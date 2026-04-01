from src.config import settings
from src.generation.base import BaseLLM


def get_llm(provider: str = None) -> BaseLLM:
    provider = (provider or settings.llm_provider).lower()

    if provider == "openai":
        from src.generation.openai_llm import OpenAILLM

        return OpenAILLM()

    elif provider == "ollama":
        from src.generation.ollama_llm import OllamaLLM

        return OllamaLLM()

    else:
        raise ValueError(f"Unknown LLM provider: '{provider}'.")
