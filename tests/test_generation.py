import pytest
from src.generation.base import BaseLLM
from tests.conftest import FakeLLM


def test_build_context_includes_text():
    llm = FakeLLM()
    chunks = [
        {"text": "Перший фрагмент тексту.", "metadata": {"filename": "doc.pdf"}},
        {"text": "Другий фрагмент тексту.", "metadata": {}},
    ]
    context = llm._build_context(chunks)
    assert "Перший фрагмент" in context
    assert "Другий фрагмент" in context


def test_build_context_numbers_chunks():
    llm = FakeLLM()
    chunks = [{"text": "А", "metadata": {}}, {"text": "Б", "metadata": {}}]
    context = llm._build_context(chunks)
    assert "[1]" in context
    assert "[2]" in context


def test_build_context_separates_chunks():
    llm = FakeLLM()
    chunks = [{"text": "А", "metadata": {}}, {"text": "Б", "metadata": {}}]
    context = llm._build_context(chunks)
    assert "---" in context


def test_build_prompt_contains_question():
    llm = FakeLLM()
    prompt = llm._build_prompt("Що таке гліфосат?", "Контекст тут.")
    assert "Що таке гліфосат?" in prompt


def test_build_prompt_contains_context():
    llm = FakeLLM()
    prompt = llm._build_prompt("Питання?", "Важливий контекст про гліфосат.")
    assert "Важливий контекст про гліфосат." in prompt


def test_build_prompt_no_leading_whitespace_per_line():
    llm = FakeLLM()
    prompt = llm._build_prompt("Q?", "ctx")
    for line in prompt.splitlines():
        assert not line.startswith("        "), f"Line has excessive indent: {line!r}"


def test_fake_llm_generate_with_chunks():
    llm = FakeLLM()
    chunks = [{"text": "текст", "metadata": {}}]
    answer = llm.generate("питання", chunks)
    assert "1 chunk" in answer


def test_fake_llm_generate_no_chunks():
    llm = FakeLLM()
    answer = llm.generate("питання", [])
    assert answer == "No context provided."
