from abc import ABC, abstractmethod
from typing import List

import structlog

log = structlog.get_logger(__name__)


class BaseLLM(ABC):
    @abstractmethod
    def generate(self, question: str, context_chunks: List[dict]) -> str: ...

    def _build_context(self, chunks: List[dict]) -> str:
        parts = []
        for i, chunk in enumerate(chunks, 1):
            source = chunk.get("source", "unknown")
            parts.append(f"[{i}] (source: {source})\n{chunk['text']}")
        return "\n\n---\n\n".join(parts)

    def _build_prompt(self, question: str, context: str) -> str:
        return (
            "You are a document Q&A assistant. Follow these rules exactly:\n\n"
            "1. Answer using ONLY information explicitly stated in the <context> below.\n"
            "2. Do NOT infer, combine facts, or add anything from outside the context.\n"
            "3. If the answer is not found in the context, respond ONLY with:\n"
            "   - Ukrainian question → 'У наданих документах немає інформації для відповіді.'\n"
            "   - Russian question  → 'В предоставленных документах нет информации для ответа.'\n"
            "   - English question  → 'The provided documents do not contain an answer.'\n"
            "4. ALWAYS respond in the SAME language as the question.\n"
            "5. Be concise. Cite only what the context says directly.\n\n"
            f"<context>\n{context}\n</context>\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )
