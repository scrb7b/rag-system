from dataclasses import dataclass
from typing import List, Optional

import structlog

from src.generation import get_llm, BaseLLM
from src.ingestion.embedder import VectorStore

log = structlog.get_logger(__name__)


@dataclass
class QAResult:
    question: str
    answer: str
    sources: List[str]
    chunks_used: List[dict]


class QAAgent:
    def __init__(self, vector_store: VectorStore, llm: Optional[BaseLLM] = None) -> None:
        self._store = vector_store
        self._llm = llm or get_llm()

    def ask(self, question: str, top_k: int = None) -> QAResult:
        log.info("Received question", question=question[:120])

        try:
            chunks = self._store.search(question, top_k=top_k)
        except Exception as exc:
            log.error("Search failed", error=str(exc))
            chunks = []

        if not chunks:
            log.warning("No relevant chunks found", question=question[:120])
            return QAResult(
                question=question,
                answer="У наданих документах немає інформації для відповіді на це питання.",
                sources=[],
                chunks_used=[],
            )

        try:
            answer = self._llm.generate(question, chunks)
        except Exception as exc:
            log.error("Generation failed", error=str(exc))
            raise

        sources = list(
            dict.fromkeys(
                c.get("metadata", {}).get("filename")
                or c.get("metadata", {}).get("source")
                or "unknown"
                for c in chunks
            )
        )

        log.info("Answer ready", sources=sources, chunks=len(chunks))

        return QAResult(
            question=question,
            answer=answer,
            sources=sources,
            chunks_used=chunks,
        )
