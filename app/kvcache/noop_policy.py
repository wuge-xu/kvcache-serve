from typing import Any

from app.kvcache.policy import (
    KVCachePolicy,
    KVCachePolicyContext,
    KVCachePolicyResult,
)


class NoOpKVCachePolicy(KVCachePolicy):
    name = "noop"

    def apply(
        self,
        past_key_values: Any,
        context: KVCachePolicyContext,
    ) -> KVCachePolicyResult:
        return self.build_result(
            past_key_values=past_key_values,
            context=context,
            tokens_after=context.cached_tokens,
            metadata={
                "cache_modified": False,
            },
        )
