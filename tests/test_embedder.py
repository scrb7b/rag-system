import pytest
from src.ingestion.embedder import VectorStore


def test_vector_store_initially_empty(vector_store: VectorStore):
    assert vector_store.count() == 0


def test_add_chunks_returns_count(vector_store: VectorStore):
    chunks = [
        {"text": "Тест перший чанк.", "metadata": {"filename": "a.pdf"}},
        {"text": "Тест другий чанк.", "metadata": {"filename": "a.pdf"}},
    ]
    added = vector_store.add_chunks(chunks)
    assert added == 2


def test_count_after_add(vector_store: VectorStore):
    chunks = [{"text": f"Чанк {i}", "metadata": {}} for i in range(3)]
    vector_store.add_chunks(chunks)
    assert vector_store.count() == 3


def test_add_empty_chunks_returns_zero(vector_store: VectorStore):
    assert vector_store.add_chunks([]) == 0
    assert vector_store.count() == 0


def test_search_returns_results(populated_store: VectorStore):
    results = populated_store.search("молекулярна маса гліфосату")
    assert len(results) > 0


def test_search_result_structure(populated_store: VectorStore):
    results = populated_store.search("гліфосат")
    for hit in results:
        assert "text" in hit
        assert "metadata" in hit
        assert "score" in hit
        assert isinstance(hit["text"], str)
        assert len(hit["text"]) > 0


def test_search_top_k_limit(populated_store: VectorStore):
    results = populated_store.search("гліфосат", top_k=2)
    assert len(results) <= 2


def test_search_relevance(populated_store: VectorStore):
    results = populated_store.search("молекулярна маса гліфосату", top_k=1)
    assert len(results) == 1
    assert "169" in results[0]["text"] or "гліфосат" in results[0]["text"].lower()


def test_reset_clears_collection(populated_store: VectorStore):
    assert populated_store.count() > 0
    populated_store.reset()
    assert populated_store.count() == 0


def test_search_empty_store_returns_empty(vector_store: VectorStore):
    results = vector_store.search("будь-який запит")
    assert results == []
