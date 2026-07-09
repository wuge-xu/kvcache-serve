from fastapi import FastAPI

from app.api.health import router as health_router
from app.api.chat import router as chat_router
from app.api.metrics import router as metrics_router
from app.api.runtime import router as runtime_router

app = FastAPI(
    title="KVCache-Serve",
    description="LLM inference service with KV Cache Runtime and observability",
    version="0.4.0",
)

app.include_router(health_router)
app.include_router(chat_router)
app.include_router(metrics_router)
app.include_router(runtime_router)


@app.get("/")
async def root():
    return {
        "service": "KVCache-Serve",
        "message": "LLM inference service is running",
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics",
        "runtime_status": "/runtime/status",
    }
