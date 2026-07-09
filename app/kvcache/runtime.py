import threading
import time
from collections import deque

from app.kvcache.state import KVCacheState


class KVCacheRuntime:
    """
    KV Cache Runtime 负责维护服务运行时状态。

    它不负责生成 token，也不负责修改 KV Cache。
    它只负责：
    1. 记录请求数量
    2. 记录活跃请求
    3. 保存最近一次 KV Cache 状态
    4. 保存最近的 KV Cache 事件
    """

    def __init__(self):
        self._lock = threading.Lock()
        self.active_request_ids: set[str] = set()
        self.total_requests = 0
        self.finished_requests = 0
        self.error_requests = 0
        self.last_state: KVCacheState | None = None
        self.events = deque(maxlen=100)

    def request_started(self, request_id: str):
        with self._lock:
            self.total_requests += 1
            self.active_request_ids.add(request_id)
            self.events.append({
                "event_type": "request_started",
                "request_id": request_id,
                "timestamp": time.time(),
            })

    def request_finished(self, request_id: str):
        with self._lock:
            self.finished_requests += 1
            self.active_request_ids.discard(request_id)
            self.events.append({
                "event_type": "request_finished",
                "request_id": request_id,
                "timestamp": time.time(),
            })

    def request_failed(self, request_id: str, error: str):
        with self._lock:
            self.error_requests += 1
            self.active_request_ids.discard(request_id)
            self.events.append({
                "event_type": "request_failed",
                "request_id": request_id,
                "error": error,
                "timestamp": time.time(),
            })

    def update_state(self, state: KVCacheState, event_type: str = "cache_updated"):
        with self._lock:
            self.last_state = state
            self.events.append({
                "event_type": event_type,
                "request_id": state.request_id,
                "cached_tokens": state.cached_tokens,
                "memory_mb": state.estimated_memory_mb,
                "timestamp": time.time(),
            })

    def get_status(self) -> dict:
        with self._lock:
            return {
                "runtime": {
                    "total_requests": self.total_requests,
                    "finished_requests": self.finished_requests,
                    "error_requests": self.error_requests,
                    "active_requests": len(self.active_request_ids),
                    "active_request_ids": list(self.active_request_ids),
                },
                "last_kv_cache_state": (
                    self.last_state.to_dict() if self.last_state else None
                ),
                "recent_events": list(self.events)[-20:],
            }


kv_cache_runtime = KVCacheRuntime()
