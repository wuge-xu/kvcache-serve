from prometheus_client import (
    CollectorRegistry,
    generate_latest,
)

from app.metrics.queue_prometheus import (
    QueueMetricsCollector,
)


class FakeQueue:
    def get_metrics_snapshot(self):
        return {
            "queue_size": 3,
            "processing_size": 2,
            "dead_letter_size": 1,
            "jobs_submitted_total": 10,
            "processing_attempts_total": 12,
            "jobs_completed_total": 7,
            "jobs_failed_total": 1,
            "retries_total": 2,
            "recoveries_total": 1,
            "dead_lettered_total": 1,
            "queue_wait_histogram": {
                "count": 2,
                "sum": 0.6,
                "buckets": [
                    {"le": "0.1", "count": 0},
                    {"le": "0.5", "count": 1},
                    {"le": "1.0", "count": 2},
                    {"le": "+Inf", "count": 2},
                ],
            },
            "inference_duration_histogram": {
                "count": 2,
                "sum": 1.5,
                "buckets": [
                    {"le": "0.5", "count": 1},
                    {"le": "1.0", "count": 2},
                    {"le": "+Inf", "count": 2},
                ],
            },
        }


class FailingQueue:
    def get_metrics_snapshot(self):
        raise RuntimeError("Redis unavailable")


def render_collector(queue) -> str:
    registry = CollectorRegistry()
    registry.register(
        QueueMetricsCollector(queue)
    )

    return generate_latest(
        registry
    ).decode("utf-8")


def test_queue_metrics():
    output = render_collector(FakeQueue())

    expected_lines = (
        "kvcache_queue_metrics_up 1.0",
        "kvcache_queue_pending_jobs 3.0",
        "kvcache_queue_processing_jobs 2.0",
        "kvcache_queue_dead_letter_jobs 1.0",
        "kvcache_worker_jobs_in_progress 2.0",
        "kvcache_queue_jobs_submitted_total 10.0",
        "kvcache_worker_processing_attempts_total 12.0",
        "kvcache_queue_jobs_completed_total 7.0",
        "kvcache_queue_jobs_failed_total 1.0",
        "kvcache_queue_jobs_retried_total 2.0",
        "kvcache_queue_jobs_recovered_total 1.0",
        "kvcache_queue_jobs_dead_lettered_total 1.0",
        (
            'kvcache_worker_job_wait_seconds_bucket'
            '{le="0.5"} 1.0'
        ),
        "kvcache_worker_job_wait_seconds_count 2.0",
        "kvcache_worker_job_wait_seconds_sum 0.6",
        (
            'kvcache_worker_inference_duration_seconds_bucket'
            '{le="1.0"} 2.0'
        ),
        (
            "kvcache_worker_inference_duration_seconds_count "
            "2.0"
        ),
        (
            "kvcache_worker_inference_duration_seconds_sum "
            "1.5"
        ),
    )

    for line in expected_lines:
        assert line in output, line

    print("[PASS] queue Prometheus metrics are correct")


def test_metrics_failure_is_safe():
    output = render_collector(FailingQueue())

    assert "kvcache_queue_metrics_up 0.0" in output
    assert "kvcache_queue_pending_jobs" not in output

    print("[PASS] Redis metrics failure is handled safely")


if __name__ == "__main__":
    test_queue_metrics()
    test_metrics_failure_is_safe()
