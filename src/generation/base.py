from abc import ABC, abstractmethod

import structlog

log = structlog.get_logger(__name__)


class BaseLLM(ABC):
    @abstractmethod
    def generate(self, question: str, context_chunks: list[dict]) -> str: ...

    def _build_context(self, chunks: list[dict]) -> str:
        parts = []
        for i, chunk in enumerate(chunks, 1):
            meta = chunk.get("metadata", {})
            source = meta.get("filename") or chunk.get("source", "unknown")
            headings = meta.get("headings") or []
            header = f"[{i}] {source}"
            if headings:
                header += f" › {' › '.join(headings)}"
            parts.append(f"{header}\n{chunk['text']}")
        return "\n\n---\n\n".join(parts)

    def _build_prompt(self, question: str, context: str) -> str:
        return (
            "You are a precise document Q&A assistant. Rules:\n\n"
            "1. Answer using ONLY facts explicitly stated in the provided context.\n"
            "2. Do NOT infer, assume, or add anything from outside the context.\n"
            "3. Respond in the SAME language as the question (Ukrainian/Russian/English).\n"
            "4. Be direct and concise — no preamble, no 'Based on the context...', no explanations.\n"
            "   Give only the factual answer, nothing else.\n"
            "5. If the answer is not in the context, reply only:\n"
            "   - Ukrainian → 'У наданих документах немає інформації для відповіді.'\n"
            "   - Russian   → 'В предоставленных документах нет информации для ответа.'\n"
            "   - English   → 'The provided documents do not contain an answer.'\n\n"
            f"<context>\n{context}\n</context>\n\n"
            f"<question>{question}</question>\n\n"
            "Answer (direct, no preamble):"
        )
