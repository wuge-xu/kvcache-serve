from dataclasses import dataclass, asdict
import time


@dataclass
class KVCacheState:
    request_id: str
    backend: str
    model_name: str
    device: str

    num_layers: int
    num_heads: int
    head_dim: int
    dtype: str
    batch_size: int

    prompt_tokens: int
    generated_tokens: int
    cached_tokens: int

    estimated_memory_bytes: int
    estimated_memory_mb: float

    created_at: float
    updated_at: float

    def to_dict(self) -> dict:
        return asdict(self)


def empty_kv_cache_state(
    request_id: str,
    backend: str,
    model_name: str,
    device: str,
) -> KVCacheState:
    now = time.time()

    return KVCacheState(
        request_id=request_id,
        backend=backend,
        model_name=model_name,
        device=device,

        num_layers=0,
        num_heads=0,
        head_dim=0,
        dtype="unknown",
        batch_size=0,

        prompt_tokens=0,
        generated_tokens=0,
        cached_tokens=0,

        estimated_memory_bytes=0,
        estimated_memory_mb=0.0,

        created_at=now,
        updated_at=now,
    )
