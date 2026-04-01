from typing import List

import structlog
from openai import OpenAI, OpenAIError
from openai.types.chat import (
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)

from src.config import settings
from src.generation.base import BaseLLM

log = structlog.get_logger(__name__)


class OpenAILLM(BaseLLM):
    def __init__(self) -> None:
        self._client = OpenAI(
            api_key=settings.openai_api_key.get_secret_value()
            if settings.openai_api_key
            else None
        )
        self._model = settings.openai_model
        log.info("OpenAI LLM ready", model=self._model)

    def generate(self, question: str, context_chunks: List[dict]) -> str:
        context = self._build_context(context_chunks)
        prompt = self._build_prompt(question, context)
        log.debug("Generating answer", model=self._model, context_chunks=len(context_chunks))
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    ChatCompletionSystemMessageParam(
                        role="system",
                        content="You are a helpful, precise Q&A assistant.",
                    ),
                    ChatCompletionUserMessageParam(role="user", content=prompt),
                ],
                temperature=settings.temperature,
                max_tokens=settings.max_tokens,
            )
            text = response.choices[0].message.content.strip()
            log.debug("Generation done", chars=len(text))
            return text
        except OpenAIError as exc:
            log.error("OpenAI API error", error=str(exc), model=self._model)
            raise
        except Exception as exc:
            log.error("Generation failed", error=str(exc))
            raise
