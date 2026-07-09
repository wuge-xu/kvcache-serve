import time

from app.inference.backend import GenerationRequest, InferenceBackend
from app.inference.result import GenerationResult


class MockBackend(InferenceBackend):
    name = "mock"

    def estimate_tokens(self, text: str) -> int:
        if not text:
            return 0
        return max(1, len(text) // 2)

    def generate(self, request: GenerationRequest) -> GenerationResult:
        start_time = time.perf_counter()

        # 模拟推理耗时
        time.sleep(0.3)

        answer = (
            "这是 KVCache-Serve 当前的 Mock LLM 回答。"
            f"你输入的问题是：{request.prompt}。"
            "当前阶段重点是保持服务结构清晰，并引入 KV Cache Runtime。"
        )

        prompt_tokens = self.estimate_tokens(request.prompt)
        completion_tokens = self.estimate_tokens(answer)
        total_tokens = prompt_tokens + completion_tokens
        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

        return GenerationResult(
            request_id=request.request_id,

            answer=answer,
            model="mock-llm",
            backend=self.name,
            device="mock",

            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,

            latency_ms=latency_ms,
            prefill_ms=0.0,
            decode_ms=0.0,
            ttft_ms=0.0,
            avg_itl_ms=0.0,
            tokens_per_second=round(completion_tokens / (latency_ms / 1000), 2),

            kv_cache_tokens=0,
            kv_cache_memory_bytes=0,
            kv_cache_memory_mb=0.0,
        )


mock_backend = MockBackend()
