from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics._faithfulness import Faithfulness
from ragas.metrics._answer_relevance import AnswerRelevancy
from ragas.metrics._context_precision import ContextPrecision
from ragas.metrics._context_recall import ContextRecall

from src.config import settings


def build_llm(provider: str) -> LangchainLLMWrapper:
    if provider == "openai":
        api_key = (
            settings.openai_api_key.get_secret_value()
            if settings.openai_api_key
            else None
        )
        return LangchainLLMWrapper(
            ChatOpenAI(model=settings.openai_model, api_key=api_key)
        )

    return LangchainLLMWrapper(
        ChatOpenAI(
            model=settings.ollama_model,
            base_url=f"{str(settings.ollama_base_url).rstrip('/')}/v1",
            api_key="ollama",
        )
    )


def build_embeddings() -> LangchainEmbeddingsWrapper:
    return LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name=settings.embed_model)
    )


def build_metrics(names: list, llm, embeddings) -> list:
    mapping = {
        "faithfulness": lambda: Faithfulness(llm=llm),
        "answer_relevancy": lambda: AnswerRelevancy(llm=llm, embeddings=embeddings),
        "context_precision": lambda: ContextPrecision(llm=llm),
        "context_recall": lambda: ContextRecall(llm=llm),
    }
    return [mapping[n]() for n in names]
