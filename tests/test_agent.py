import pytest
from unittest.mock import MagicMock, patch
from src.agent import QAAgent, QAResult
from tests.conftest import FakeLLM


def test_agent_returns_qa_result(populated_store, fake_llm):
    agent = QAAgent(populated_store, fake_llm)
    result = agent.ask("Яка молекулярна маса гліфосату?")
    assert isinstance(result, QAResult)


def test_agent_answer_not_empty(populated_store, fake_llm):
    agent = QAAgent(populated_store, fake_llm)
    result = agent.ask("Яка молекулярна маса гліфосату?")
    assert result.answer
    assert len(result.answer) > 0


def test_agent_populates_sources(populated_store, fake_llm):
    agent = QAAgent(populated_store, fake_llm)
    result = agent.ask("гліфосат")
    assert len(result.sources) > 0
    for src in result.sources:
        assert isinstance(src, str)


def test_agent_populates_chunks_used(populated_store, fake_llm):
    agent = QAAgent(populated_store, fake_llm)
    result = agent.ask("гліфосат")
    assert len(result.chunks_used) > 0


def test_agent_no_chunks_gives_fallback(vector_store, fake_llm):
    """Empty store → agent returns the 'no info' fallback message."""
    agent = QAAgent(vector_store, fake_llm)
    result = agent.ask("Щось дуже специфічне чого немає в базі")
    assert result.answer
    assert result.chunks_used == []
    assert result.sources == []


def test_agent_question_preserved(populated_store, fake_llm):
    question = "Яка норма витрати препарату Акріс?"
    agent = QAAgent(populated_store, fake_llm)
    result = agent.ask(question)
    assert result.question == question


def test_agent_uses_default_llm_when_none(populated_store):
    """QAAgent creates its own LLM if not provided — just check no crash on init."""
    with patch("src.agent.get_llm") as mock_get_llm:
        mock_get_llm.return_value = FakeLLM()
        agent = QAAgent(populated_store)
        assert agent._llm is not None
