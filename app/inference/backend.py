from abc import ABC, abstractmethod
from dataclasses import dataclass

from app.inference.result import GenerationResult


@dataclass
class GenerationRequest:
    request_id: str
    prompt: str
    system_prompt: str | None = None
    max_tokens: int = 64


class InferenceBackend(ABC):
    name: str

    @abstractmethod
    def generate(self, request: GenerationRequest) -> GenerationResult:
        raise NotImplementedError
