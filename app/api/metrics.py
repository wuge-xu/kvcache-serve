from fastapi import APIRouter, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    generate_latest,
)

from app.metrics.queue_prometheus import (
    register_queue_metrics,
)


router = APIRouter()

register_queue_metrics()


@router.get("/metrics")
async def metrics():
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )
