from prometheus_client.core import (
    CounterMetricFamily,
    GaugeMetricFamily,
    HistogramMetricFamily,
)
from prometheus_client.registry import REGISTRY

from app.scheduler.redis_queue import redis_queue


class QueueMetricsCollector:
    """Export Redis-backed queue statistics to Prometheus."""

    def __init__(self, queue=redis_queue):
        self.queue = queue

    def collect(self):
        metrics_up = GaugeMetricFamily(
            "kvcache_queue_metrics_up",
            "Whether queue metrics could be read from Redis",
        )

        try:
            snapshot = self.queue.get_metrics_snapshot()
        except Exception:
            metrics_up.add_metric([], 0)
            yield metrics_up
            return

        metrics_up.add_metric([], 1)
        yield metrics_up

        gauges = (
            (
                "kvcache_queue_pending_jobs",
                "Current number of pending queue jobs",
                snapshot["queue_size"],
            ),
            (
                "kvcache_queue_processing_jobs",
                "Current number of jobs being processed",
                snapshot["processing_size"],
            ),
            (
                "kvcache_queue_dead_letter_jobs",
                "Current number of jobs in the dead-letter queue",
                snapshot["dead_letter_size"],
            ),
            (
                "kvcache_worker_jobs_in_progress",
                "Current number of worker jobs in progress",
                snapshot["processing_size"],
            ),
        )

        for name, description, value in gauges:
            metric = GaugeMetricFamily(
                name,
                description,
            )
            metric.add_metric([], value)
            yield metric

        counters = (
            (
                "kvcache_queue_jobs_submitted",
                "Total number of submitted queue jobs",
                snapshot["jobs_submitted_total"],
            ),
            (
                "kvcache_worker_processing_attempts",
                "Total number of worker processing attempts",
                snapshot["processing_attempts_total"],
            ),
            (
                "kvcache_queue_jobs_completed",
                "Total number of completed queue jobs",
                snapshot["jobs_completed_total"],
            ),
            (
                "kvcache_queue_jobs_failed",
                "Total number of permanently failed queue jobs",
                snapshot["jobs_failed_total"],
            ),
            (
                "kvcache_queue_jobs_retried",
                "Total number of queue retries",
                snapshot["retries_total"],
            ),
            (
                "kvcache_queue_jobs_recovered",
                "Total number of stale processing jobs recovered",
                snapshot["recoveries_total"],
            ),
            (
                "kvcache_queue_jobs_dead_lettered",
                "Total number of jobs moved to the dead-letter queue",
                snapshot["dead_lettered_total"],
            ),
        )

        for name, description, value in counters:
            metric = CounterMetricFamily(
                name,
                description,
            )
            metric.add_metric([], value)
            yield metric

        yield self._histogram_metric(
            name="kvcache_worker_job_wait_seconds",
            description=(
                "Time between queue insertion and worker processing"
            ),
            snapshot=snapshot["queue_wait_histogram"],
        )

        yield self._histogram_metric(
            name="kvcache_worker_inference_duration_seconds",
            description="Worker inference execution duration",
            snapshot=snapshot[
                "inference_duration_histogram"
            ],
        )

    @staticmethod
    def _histogram_metric(
        name: str,
        description: str,
        snapshot: dict,
    ) -> HistogramMetricFamily:
        metric = HistogramMetricFamily(
            name,
            description,
        )

        buckets = [
            (
                str(item["le"]),
                int(item["count"]),
            )
            for item in snapshot["buckets"]
        ]

        metric.add_metric(
            [],
            buckets,
            float(snapshot["sum"]),
        )

        return metric


QUEUE_METRICS_COLLECTOR = QueueMetricsCollector()


def register_queue_metrics(
    registry=REGISTRY,
):
    try:
        registry.register(
            QUEUE_METRICS_COLLECTOR
        )
    except ValueError:
        # Module reloads during development can attempt
        # to register the same collector more than once.
        pass
