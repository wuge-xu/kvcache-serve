import time
from dataclasses import dataclass

import torch

from app.config.settings import DEFAULT_MAX_NEW_TOKENS
from app.inference.model_loader import model_loader


@dataclass
class GenerationResult:
    answer: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float
    device: str

    prefill_ms: float
    decode_ms: float
    ttft_ms: float
    avg_itl_ms: float
    tokens_per_second: float

    kv_cache_tokens: int
    kv_cache_memory_bytes: int
    kv_cache_memory_mb: float


class LocalLLMGenerator:
    """
    本地 Transformers 推理器。

    当前版本不再直接使用 model.generate，
    而是手动拆分：
    1. Prefill：处理完整 prompt，生成 past_key_values
    2. Decode：每次输入 1 个 token，复用 past_key_values
    """

    def build_prompt(self, prompt: str, system_prompt: str | None = None) -> str:
        if system_prompt:
            return f"System: {system_prompt}\nUser: {prompt}\nAssistant:"
        return f"User: {prompt}\nAssistant:"

    def _past_to_legacy(self, past_key_values):
        """
        兼容不同 transformers 版本。
        有些版本 past_key_values 是 tuple，
        有些版本可能是 Cache 对象。
        """
        if past_key_values is None:
            return None

        if hasattr(past_key_values, "to_legacy_cache"):
            return past_key_values.to_legacy_cache()

        return past_key_values

    def estimate_kv_cache(self, past_key_values) -> tuple[int, int]:
        """
        估算 KV Cache 中 token 数和显存/内存占用。

        对 GPT 类模型来说，KV Cache 通常形状类似：
        key:   [batch, num_heads, seq_len, head_dim]
        value: [batch, num_heads, seq_len, head_dim]

        memory bytes = key tensor bytes + value tensor bytes
        """
        if past_key_values is None:
            return 0, 0

        # 新版 Cache 对象可能支持 get_seq_length
        kv_tokens = 0
        if hasattr(past_key_values, "get_seq_length"):
            try:
                kv_tokens = int(past_key_values.get_seq_length())
            except Exception:
                kv_tokens = 0

        legacy_cache = self._past_to_legacy(past_key_values)

        total_bytes = 0
        try:
            for layer_cache in legacy_cache:
                if not isinstance(layer_cache, (tuple, list)) or len(layer_cache) < 2:
                    continue

                key_tensor = layer_cache[0]
                value_tensor = layer_cache[1]

                total_bytes += key_tensor.numel() * key_tensor.element_size()
                total_bytes += value_tensor.numel() * value_tensor.element_size()

                if kv_tokens == 0 and key_tensor.dim() >= 3:
                    kv_tokens = int(key_tensor.shape[-2])

        except Exception:
            return kv_tokens, total_bytes

        return kv_tokens, total_bytes

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_new_tokens: int | None = None,
    ) -> GenerationResult:
        tokenizer, model, device, model_name = model_loader.get()

        max_new_tokens = max_new_tokens or DEFAULT_MAX_NEW_TOKENS
        full_prompt = self.build_prompt(prompt, system_prompt)

        inputs = tokenizer(full_prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")

        prompt_tokens = input_ids.shape[-1]

        request_start = time.perf_counter()

        generated_token_ids = []
        decode_step_latencies = []

        with torch.inference_mode():
            # 1. Prefill 阶段：一次性处理完整 prompt，生成 KV Cache
            prefill_start = time.perf_counter()

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
            )

            past_key_values = outputs.past_key_values

            # 从 prompt 最后一个位置预测第一个新 token
            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            prefill_end = time.perf_counter()

            generated_token_ids.append(int(next_token.item()))

            ttft_ms = round((prefill_end - request_start) * 1000, 2)
            prefill_ms = round((prefill_end - prefill_start) * 1000, 2)

            # 2. Decode 阶段：每次输入上一个 token，复用 KV Cache
            decode_start = time.perf_counter()

            for _ in range(max_new_tokens - 1):
                step_start = time.perf_counter()

                outputs = model(
                    input_ids=next_token,
                    past_key_values=past_key_values,
                    use_cache=True,
                )

                past_key_values = outputs.past_key_values

                next_token_logits = outputs.logits[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                step_end = time.perf_counter()
                decode_step_latencies.append(step_end - step_start)

                token_id = int(next_token.item())
                generated_token_ids.append(token_id)

                if token_id == tokenizer.eos_token_id:
                    break

            decode_end = time.perf_counter()

        total_latency_ms = round((time.perf_counter() - request_start) * 1000, 2)
        decode_ms = round((decode_end - decode_start) * 1000, 2)

        completion_tokens = len(generated_token_ids)
        total_tokens = prompt_tokens + completion_tokens

        if decode_step_latencies:
            avg_itl_ms = round((sum(decode_step_latencies) / len(decode_step_latencies)) * 1000, 2)
        else:
            avg_itl_ms = 0.0

        if total_latency_ms > 0:
            tokens_per_second = round(completion_tokens / (total_latency_ms / 1000), 2)
        else:
            tokens_per_second = 0.0

        output_tensor = torch.tensor(generated_token_ids, device=device)
        answer = tokenizer.decode(output_tensor, skip_special_tokens=True).strip()

        if not answer:
            answer = "(模型没有生成可见文本)"

        kv_cache_tokens, kv_cache_memory_bytes = self.estimate_kv_cache(past_key_values)
        kv_cache_memory_mb = round(kv_cache_memory_bytes / 1024 / 1024, 4)

        return GenerationResult(
            answer=answer,
            model=model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency_ms=total_latency_ms,
            device=device,

            prefill_ms=prefill_ms,
            decode_ms=decode_ms,
            ttft_ms=ttft_ms,
            avg_itl_ms=avg_itl_ms,
            tokens_per_second=tokens_per_second,

            kv_cache_tokens=kv_cache_tokens,
            kv_cache_memory_bytes=kv_cache_memory_bytes,
            kv_cache_memory_mb=kv_cache_memory_mb,
        )


local_generator = LocalLLMGenerator()
