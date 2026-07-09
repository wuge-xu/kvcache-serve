import time

import torch

from app.config.settings import DEFAULT_MAX_NEW_TOKENS
from app.inference.backend import GenerationRequest, InferenceBackend
from app.inference.result import GenerationResult
from app.inference.model_loader import model_loader
from app.kvcache.estimator import kv_cache_estimator
from app.kvcache.runtime import kv_cache_runtime


class TransformersBackend(InferenceBackend):
    name = "transformers"

    def build_prompt(self, prompt: str, system_prompt: str | None = None) -> str:
        if system_prompt:
            return f"System: {system_prompt}\nUser: {prompt}\nAssistant:"
        return f"User: {prompt}\nAssistant:"

    def generate(self, request: GenerationRequest) -> GenerationResult:
        tokenizer, model, device, model_name = model_loader.get()

        max_new_tokens = request.max_tokens or DEFAULT_MAX_NEW_TOKENS
        full_prompt = self.build_prompt(request.prompt, request.system_prompt)

        inputs = tokenizer(full_prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")

        prompt_tokens = input_ids.shape[-1]

        request_start = time.perf_counter()

        generated_token_ids = []
        decode_step_latencies = []

        with torch.inference_mode():
            # 1. Prefill：处理完整 prompt，生成 KV Cache
            prefill_start = time.perf_counter()

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
            )

            past_key_values = outputs.past_key_values

            prefill_end = time.perf_counter()

            prefill_state = kv_cache_estimator.estimate(
                past_key_values=past_key_values,
                request_id=request.request_id,
                backend=self.name,
                model_name=model_name,
                device=device,
                prompt_tokens=prompt_tokens,
                generated_tokens=0,
                model_config=model.config,
                model_dtype=str(next(model.parameters()).dtype),
            )
            kv_cache_runtime.update_state(prefill_state, event_type="cache_created")

            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            generated_token_ids.append(int(next_token.item()))

            ttft_ms = round((prefill_end - request_start) * 1000, 2)
            prefill_ms = round((prefill_end - prefill_start) * 1000, 2)

            # 2. Decode：每次生成一个 token，复用 KV Cache
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
            avg_itl_ms = round(
                (sum(decode_step_latencies) / len(decode_step_latencies)) * 1000,
                2,
            )
        else:
            avg_itl_ms = 0.0

        if total_latency_ms > 0:
            tokens_per_second = round(completion_tokens / (total_latency_ms / 1000), 2)
        else:
            tokens_per_second = 0.0

        answer = tokenizer.decode(
            generated_token_ids,
            skip_special_tokens=True,
        ).strip()

        if not answer:
            answer = "(模型没有生成可见文本)"

        final_state = kv_cache_estimator.estimate(
            past_key_values=past_key_values,
            request_id=request.request_id,
            backend=self.name,
            model_name=model_name,
            device=device,
            prompt_tokens=prompt_tokens,
            generated_tokens=completion_tokens,
            model_config=model.config,
            model_dtype=str(next(model.parameters()).dtype),
        )
        kv_cache_runtime.update_state(final_state, event_type="cache_updated")

        return GenerationResult(
            request_id=request.request_id,

            answer=answer,
            model=model_name,
            backend=self.name,
            device=device,

            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,

            latency_ms=total_latency_ms,
            prefill_ms=prefill_ms,
            decode_ms=decode_ms,
            ttft_ms=ttft_ms,
            avg_itl_ms=avg_itl_ms,
            tokens_per_second=tokens_per_second,

            kv_cache_tokens=final_state.cached_tokens,
            kv_cache_memory_bytes=final_state.estimated_memory_bytes,
            kv_cache_memory_mb=final_state.estimated_memory_mb,
        )


transformers_backend = TransformersBackend()
