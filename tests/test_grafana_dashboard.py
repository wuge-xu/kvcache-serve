import json
from pathlib import Path


DASHBOARD_PATH = Path(
    "deploy/compose/grafana/dashboards/"
    "kvcache-queue-reliability.json"
)


def test_grafana_dashboard():
    dashboard = json.loads(
        DASHBOARD_PATH.read_text(
            encoding="utf-8",
        )
    )

    assert (
        dashboard["uid"]
        == "kvcache-queue-reliability"
    )

    assert (
        dashboard["title"]
        == "KVCache-Serve Queue Reliability"
    )

    panels = dashboard["panels"]

    assert len(panels) == 16

    panel_ids = [
        panel["id"]
        for panel in panels
    ]

    assert len(panel_ids) == len(
        set(panel_ids)
    )

    expressions = []

    for panel in panels:
        grid = panel["gridPos"]

        assert grid["w"] > 0
        assert grid["h"] > 0
        assert 0 <= grid["x"] < 24
        assert grid["x"] + grid["w"] <= 24

        for target in panel.get(
            "targets",
            [],
        ):
            expression = target.get(
                "expr",
                "",
            )

            if expression:
                expressions.append(expression)

    combined = "\n".join(expressions)

    required_metrics = (
        "kvcache_queue_metrics_up",
        "kvcache_queue_pending_jobs",
        "kvcache_queue_processing_jobs",
        "kvcache_queue_dead_letter_jobs",
        "kvcache_queue_jobs_submitted_total",
        "kvcache_queue_jobs_completed_total",
        "kvcache_queue_jobs_failed_total",
        "kvcache_queue_jobs_retried_total",
        "kvcache_queue_jobs_recovered_total",
        "kvcache_queue_jobs_dead_lettered_total",
        "kvcache_worker_processing_attempts_total",
        "kvcache_worker_job_wait_seconds_bucket",
        (
            "kvcache_worker_"
            "inference_duration_seconds_bucket"
        ),
    )

    for metric in required_metrics:
        assert metric in combined, metric

    variables = dashboard[
        "templating"
    ]["list"]

    assert any(
        variable.get("name")
        == "DS_PROMETHEUS"
        for variable in variables
    )

    print(
        "[PASS] Grafana reliability "
        "dashboard is valid"
    )

    print(
        {
            "title": dashboard["title"],
            "uid": dashboard["uid"],
            "panels": len(panels),
            "queries": len(expressions),
        }
    )


if __name__ == "__main__":
    test_grafana_dashboard()
