from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from app.inference.result import GenerationResult


@dataclass
class GenerationRequest:
    request_id: str
    prompt: str
    system_prompt: str | None = None
    max_tokens: int = 64

    kv_policy: str = "noop"
    kv_policy_config: dict[str, Any] = field(
        default_factory=dict,
    )


class InferenceBackend(ABC):
    name: str

    @abstractmethod
    def generate(
        self,
        request: GenerationRequest,
    ) -> GenerationResult:
        raise NotImplementedError
