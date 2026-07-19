from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class KVCachePolicyContext:
    request_id: str
    stage: str
    decode_step: int

    prompt_tokens: int
    generated_tokens: int
    cached_tokens: int

    model_name: str
    device: str


@dataclass
class KVCachePolicyResult:
    past_key_values: Any

    request_id: str
    policy_name: str
    stage: str
    decode_step: int

    tokens_before: int
    tokens_after: int
    evicted_tokens: int

    metadata: dict[str, Any] = field(
        default_factory=dict,
    )


class KVCachePolicy(ABC):
    name: str

    def __init__(
        self,
        config: dict[str, Any] | None = None,
    ):
        self.config = dict(config or {})

    @abstractmethod
    def apply(
        self,
        past_key_values: Any,
        context: KVCachePolicyContext,
    ) -> KVCachePolicyResult:
        raise NotImplementedError

    def build_result(
        self,
        past_key_values: Any,
        context: KVCachePolicyContext,
        tokens_after: int,
        metadata: dict[str, Any] | None = None,
    ) -> KVCachePolicyResult:
        tokens_before = max(
            0,
            int(context.cached_tokens),
        )

        tokens_after = max(
            0,
            int(tokens_after),
        )

        return KVCachePolicyResult(
            past_key_values=past_key_values,
            request_id=context.request_id,
            policy_name=self.name,
            stage=context.stage,
            decode_step=context.decode_step,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            evicted_tokens=max(
                0,
                tokens_before - tokens_after,
            ),
            metadata=dict(metadata or {}),
        )
