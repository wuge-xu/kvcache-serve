import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
K8S_DIR = ROOT / "deploy" / "k8s"


def read_file(name: str) -> str:
    return (K8S_DIR / name).read_text(
        encoding="utf-8",
    )


def require_tokens(
    content: str,
    tokens: tuple[str, ...],
    filename: str,
) -> None:
    for token in tokens:
        assert token in content, (
            f"{filename} missing token: {token}"
        )


def test_k8s_reliable_queue_manifests():
    kustomization = read_file(
        "kustomization.yaml"
    )

    for resource in (
        "namespace.yaml",
        "config.yaml",
        "redis.yaml",
        "api.yaml",
        "worker.yaml",
        "reaper.yaml",
    ):
        assert f"- {resource}" in kustomization

    config = read_file("config.yaml")

    require_tokens(
        config,
        (
            "name: kvcache-config",
            "REDIS_HOST: kvcache-redis",
            'REDIS_PORT: "6379"',
            'QUEUE_MAX_RETRIES: "2"',
            (
                "PROCESSING_TIMEOUT_SECONDS: "
                '"30"'
            ),
            'REAPER_INTERVAL_SECONDS: "5"',
            'MAX_RECOVERIES: "2"',
        ),
        "config.yaml",
    )

    redis = read_file("redis.yaml")

    require_tokens(
        redis,
        (
            "kind: PersistentVolumeClaim",
            "name: kvcache-redis-data",
            "storageClassName: local-path",
            "storage: 1Gi",
            "strategy:",
            "type: Recreate",
            "image: redis:7-alpine",
            "--appendonly",
            "--appendfsync",
            "everysec",
            "mountPath: /data",
            "claimName: kvcache-redis-data",
            "readinessProbe:",
            "livenessProbe:",
        ),
        "redis.yaml",
    )

    api = read_file("api.yaml")

    require_tokens(
        api,
        (
            "name: kvcache-api",
            "image: kvcache-serve:local",
            "imagePullPolicy: Never",
            "name: kvcache-config",
            'prometheus.io/scrape: "true"',
            "prometheus.io/path: /metrics",
            "startupProbe:",
            "readinessProbe:",
            "livenessProbe:",
        ),
        "api.yaml",
    )

    worker = read_file("worker.yaml")

    require_tokens(
        worker,
        (
            "name: kvcache-worker",
            "image: kvcache-serve:local",
            "imagePullPolicy: Never",
            "app.worker",
            "name: kvcache-config",
            "terminationGracePeriodSeconds: 10",
        ),
        "worker.yaml",
    )

    reaper = read_file("reaper.yaml")

    require_tokens(
        reaper,
        (
            "name: kvcache-reaper",
            "image: kvcache-serve:local",
            "imagePullPolicy: Never",
            "app.reaper",
            "name: kvcache-config",
            "terminationGracePeriodSeconds: 10",
        ),
        "reaper.yaml",
    )

    kubectl = shutil.which("kubectl")

    if kubectl:
        result = subprocess.run(
            [
                kubectl,
                "kustomize",
                str(K8S_DIR),
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 0, (
            result.stderr
        )

        rendered = result.stdout

        require_tokens(
            rendered,
            (
                "name: kvcache-config",
                "name: kvcache-redis-data",
                "name: kvcache-api",
                "name: kvcache-worker",
                "name: kvcache-reaper",
                "image: kvcache-serve:local",
            ),
            "rendered manifests",
        )

    print(
        "[PASS] Kubernetes reliable queue "
        "manifests are valid"
    )


if __name__ == "__main__":
    test_k8s_reliable_queue_manifests()
