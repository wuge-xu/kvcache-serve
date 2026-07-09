from fastapi import APIRouter

from app.kvcache.runtime import kv_cache_runtime
from app.inference.model_loader import model_loader

router = APIRouter(prefix="/runtime", tags=["runtime"])


@router.get("/status")
async def runtime_status():
    model_info = {
        "configured_model": model_loader.model_name,
        "loaded": model_loader._loaded,
        "device": model_loader.device,
    }

    return {
        "service": "kvcache-serve",
        "model": model_info,
        **kv_cache_runtime.get_status(),
    }
