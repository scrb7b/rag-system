import pytest
from unittest.mock import MagicMock, patch
from src.ingestion.embedder import VectorStore
from src.generation.base import BaseLLM
from src.config import settings


@pytest.fixture(autouse=True)
def disable_reranking_for_tests(monkeypatch):
    monkeypatch.setattr(settings, "enable_reranking", False)


class FakeLLM(BaseLLM):
    def generate(self, question: str, context_chunks: list) -> str:
        if not context_chunks:
            return "No context provided."
        return f"Answer based on {len(context_chunks)} chunk(s)."


@pytest.fixture()
def fake_llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture()
def vector_store() -> VectorStore:
    return VectorStore()


@pytest.fixture()
def populated_store(vector_store: VectorStore) -> VectorStore:
    chunks = [
        {
            "text": "Молекулярна маса гліфосату становить 169.1.",
            "metadata": {"filename": "glyphosate.html"},
        },
        {
            "text": "Гліфосат відноситься до фосфорорганічних сполук.",
            "metadata": {"filename": "glyphosate.html"},
        },
        {
            "text": "Розчинність гліфосату у воді при 25°C становить 12 г/л.",
            "metadata": {"filename": "glyphosate.html"},
        },
        {
            "text": "Кукурудзу розміщують після озимих та ярих зернових культур.",
            "metadata": {"filename": "corn.pdf"},
        },
        {
            "text": "Норма витрати препарату Акріс® становить 1.5–3.0 л/га.",
            "metadata": {"filename": "corn.pdf"},
        },
    ]
    vector_store.add_chunks(chunks)
    return vector_store
