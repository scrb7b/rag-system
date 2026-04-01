from openai import OpenAI
from ragas.llms import llm_factory
from ragas.embeddings import HuggingFaceEmbeddings
from ragas.metrics.collections import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)

from src.config import settings


def build_llm(provider: str):
    if provider == "openai":
        client = OpenAI(
            api_key=settings.openai_api_key.get_secret_value() if settings.openai_api_key else "no-key",
        )
        return llm_factory(settings.openai_model, client=client)

    client = OpenAI(
        base_url=f"{str(settings.ollama_base_url).rstrip('/')}/v1",
        api_key="ollama",
    )
    return llm_factory(settings.ollama_model, client=client)


def build_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model=settings.embed_model)


def build_metrics(names: list, llm, embeddings) -> list:
    mapping = {
        "faithfulness": lambda: Faithfulness(llm=llm),
        "answer_relevancy": lambda: AnswerRelevancy(llm=llm, embeddings=embeddings),
        "context_precision": lambda: ContextPrecision(llm=llm),
        "context_recall": lambda: ContextRecall(llm=llm),
    }
    return [mapping[n]() for n in names]
