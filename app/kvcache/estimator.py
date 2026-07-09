import time
from typing import Any

from app.kvcache.state import KVCacheState, empty_kv_cache_state


class KVCacheEstimator:
    """
    负责从 past_key_values 中估算 KV Cache 状态。

    当前设计原则：
    1. 优先从真实 KV Cache tensor 中估算 token 数和 memory
    2. 兼容 tuple / list legacy cache
    3. 兼容部分新版 Transformers Cache 对象
    4. 如果真实 tensor 解析失败，则使用 model.config 做兜底估算

    注意：
    这里不修改 KV Cache，只做观测和估算。
    """

    def to_legacy_cache(self, past_key_values: Any):
        if past_key_values is None:
            return None

        if hasattr(past_key_values, "to_legacy_cache"):
            try:
                return past_key_values.to_legacy_cache()
            except Exception:
                return past_key_values

        return past_key_values

    def is_tensor_like(self, value: Any) -> bool:
        return (
            hasattr(value, "shape")
            and hasattr(value, "numel")
            and hasattr(value, "element_size")
        )

    def get_attr_if_exists(self, obj: Any, names: list[str]):
        for name in names:
            if hasattr(obj, name):
                value = getattr(obj, name)
                if value is not None:
                    return value
        return None

    def extract_kv_pairs(self, past_key_values: Any):
        """
        尝试从不同格式的 past_key_values 中提取 [(key_tensor, value_tensor), ...]
        """

        if past_key_values is None:
            return []

        pairs = []

        # 1. legacy tuple / list cache:
        # ((key, value), (key, value), ...)
        legacy_cache = self.to_legacy_cache(past_key_values)

        if isinstance(legacy_cache, (tuple, list)):
            for layer_cache in legacy_cache:
                if not isinstance(layer_cache, (tuple, list)) or len(layer_cache) < 2:
                    continue

                key_tensor = layer_cache[0]
                value_tensor = layer_cache[1]

                if self.is_tensor_like(key_tensor) and self.is_tensor_like(value_tensor):
                    pairs.append((key_tensor, value_tensor))

            if pairs:
                return pairs

        # 2. 有些 Cache 对象有 key_cache / value_cache
        key_cache = self.get_attr_if_exists(past_key_values, ["key_cache", "_key_cache"])
        value_cache = self.get_attr_if_exists(past_key_values, ["value_cache", "_value_cache"])

        if key_cache is not None and value_cache is not None:
            try:
                for key_tensor, value_tensor in zip(key_cache, value_cache):
                    if self.is_tensor_like(key_tensor) and self.is_tensor_like(value_tensor):
                        pairs.append((key_tensor, value_tensor))
            except Exception:
                pass

            if pairs:
                return pairs

        # 3. 有些新版 Cache 对象内部是 layers
        layers = self.get_attr_if_exists(past_key_values, ["layers", "_layers"])

        if layers is not None:
            try:
                for layer in layers:
                    key_tensor = self.get_attr_if_exists(
                        layer,
                        ["keys", "key", "key_cache", "_keys", "_key_cache"],
                    )
                    value_tensor = self.get_attr_if_exists(
                        layer,
                        ["values", "value", "value_cache", "_values", "_value_cache"],
                    )

                    if self.is_tensor_like(key_tensor) and self.is_tensor_like(value_tensor):
                        pairs.append((key_tensor, value_tensor))
            except Exception:
                pass

            if pairs:
                return pairs

        return []

    def dtype_to_bytes(self, dtype: str) -> int:
        dtype = str(dtype).lower()

        if "float16" in dtype or "bfloat16" in dtype or "int16" in dtype:
            return 2

        if "float32" in dtype or "int32" in dtype:
            return 4

        if "float64" in dtype or "int64" in dtype:
            return 8

        if "int8" in dtype or "uint8" in dtype:
            return 1

        # PyTorch CPU 默认多数是 float32
        return 4

    def get_config_value(self, config: Any, names: list[str], default: int = 0) -> int:
        if config is None:
            return default

        for name in names:
            if hasattr(config, name):
                value = getattr(config, name)
                if value is not None:
                    try:
                        return int(value)
                    except Exception:
                        pass

        return default

    def estimate_from_config(
        self,
        request_id: str,
        backend: str,
        model_name: str,
        device: str,
        prompt_tokens: int,
        generated_tokens: int,
        model_config: Any = None,
        model_dtype: str = "unknown",
    ) -> KVCacheState:
        """
        兜底估算。

        KV Cache 近似公式：
        memory = layers * 2(K/V) * batch * heads * cached_tokens * head_dim * dtype_bytes
        """

        num_layers = self.get_config_value(
            model_config,
            ["num_hidden_layers", "n_layer", "num_layers"],
            0,
        )

        num_heads = self.get_config_value(
            model_config,
            ["num_attention_heads", "n_head", "num_heads"],
            0,
        )

        hidden_size = self.get_config_value(
            model_config,
            ["hidden_size", "n_embd", "d_model"],
            0,
        )

        if num_heads > 0 and hidden_size > 0:
            head_dim = hidden_size // num_heads
        else:
            head_dim = 0

        batch_size = 1

        if generated_tokens <= 0:
            cached_tokens = prompt_tokens
        else:
            cached_tokens = max(0, prompt_tokens + generated_tokens - 1)

        dtype_bytes = self.dtype_to_bytes(model_dtype)

        estimated_memory_bytes = (
            num_layers
            * 2
            * batch_size
            * num_heads
            * cached_tokens
            * head_dim
            * dtype_bytes
        )

        now = time.time()

        return KVCacheState(
            request_id=request_id,
            backend=backend,
            model_name=model_name,
            device=device,

            num_layers=num_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            dtype=str(model_dtype),
            batch_size=batch_size,

            prompt_tokens=prompt_tokens,
            generated_tokens=generated_tokens,
            cached_tokens=cached_tokens,

            estimated_memory_bytes=estimated_memory_bytes,
            estimated_memory_mb=round(estimated_memory_bytes / 1024 / 1024, 6),

            created_at=now,
            updated_at=now,
        )

    def estimate(
        self,
        past_key_values: Any,
        request_id: str,
        backend: str,
        model_name: str,
        device: str,
        prompt_tokens: int,
        generated_tokens: int,
        model_config: Any = None,
        model_dtype: str = "unknown",
    ) -> KVCacheState:
        if past_key_values is None:
            return empty_kv_cache_state(
                request_id=request_id,
                backend=backend,
                model_name=model_name,
                device=device,
            )

        pairs = self.extract_kv_pairs(past_key_values)

        # 优先使用真实 tensor 统计
        if pairs:
            first_key = pairs[0][0]

            batch_size = 0
            num_heads = 0
            cached_tokens = 0
            head_dim = 0
            dtype = str(getattr(first_key, "dtype", "unknown"))

            if len(first_key.shape) >= 4:
                batch_size = int(first_key.shape[0])
                num_heads = int(first_key.shape[1])
                cached_tokens = int(first_key.shape[-2])
                head_dim = int(first_key.shape[-1])
            elif len(first_key.shape) >= 3:
                batch_size = int(first_key.shape[0])
                cached_tokens = int(first_key.shape[-2])
                head_dim = int(first_key.shape[-1])

            total_bytes = 0

            for key_tensor, value_tensor in pairs:
                total_bytes += key_tensor.numel() * key_tensor.element_size()
                total_bytes += value_tensor.numel() * value_tensor.element_size()

            now = time.time()

            return KVCacheState(
                request_id=request_id,
                backend=backend,
                model_name=model_name,
                device=device,

                num_layers=len(pairs),
                num_heads=num_heads,
                head_dim=head_dim,
                dtype=dtype,
                batch_size=batch_size,

                prompt_tokens=prompt_tokens,
                generated_tokens=generated_tokens,
                cached_tokens=cached_tokens,

                estimated_memory_bytes=total_bytes,
                estimated_memory_mb=round(total_bytes / 1024 / 1024, 6),

                created_at=now,
                updated_at=now,
            )

        # 如果真实 tensor 解析失败，用模型配置兜底估算
        return self.estimate_from_config(
            request_id=request_id,
            backend=backend,
            model_name=model_name,
            device=device,
            prompt_tokens=prompt_tokens,
            generated_tokens=generated_tokens,
            model_config=model_config,
            model_dtype=model_dtype,
        )


kv_cache_estimator = KVCacheEstimator()
