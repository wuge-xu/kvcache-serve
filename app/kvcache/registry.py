from typing import Any

from app.kvcache.noop_policy import (
    NoOpKVCachePolicy,
)
from app.kvcache.policy import KVCachePolicy


class KVCachePolicyRegistry:
    def __init__(self):
        self._policies: dict[
            str,
            type[KVCachePolicy],
        ] = {}

    @staticmethod
    def normalize_name(name: str | None) -> str:
        normalized = str(
            name or "noop"
        ).strip().lower()

        normalized = normalized.replace(
            "-",
            "_",
        )

        return normalized or "noop"

    def register(
        self,
        policy_type: type[KVCachePolicy],
    ) -> None:
        name = self.normalize_name(
            policy_type.name
        )

        self._policies[name] = policy_type

    def create(
        self,
        name: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> KVCachePolicy:
        normalized = self.normalize_name(name)
        policy_type = self._policies.get(
            normalized
        )

        if policy_type is None:
            available = ", ".join(
                self.available()
            )

            raise ValueError(
                "unknown KV Cache policy: "
                f"{normalized}; "
                f"available policies: {available}"
            )

        return policy_type(config=config)

    def available(self) -> list[str]:
        return sorted(self._policies)


kv_cache_policy_registry = (
    KVCachePolicyRegistry()
)

kv_cache_policy_registry.register(
    NoOpKVCachePolicy
)
