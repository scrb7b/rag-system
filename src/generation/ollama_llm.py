import requests
import structlog

from src.config import settings
from src.generation.base import BaseLLM

log = structlog.get_logger(__name__)


class OllamaLLM(BaseLLM):
    def __init__(self) -> None:
        self._base_url = str(settings.ollama_base_url).rstrip("/")
        self._model = settings.ollama_model
        self._check_connection()

    def _check_connection(self) -> None:
        try:
            r = requests.get(f"{self._base_url}/api/tags", timeout=5)
            r.raise_for_status()
            log.info("Connected to Ollama", url=self._base_url, model=self._model)
        except Exception as exc:
            raise RuntimeError(f"Cannot connect to Ollama at {self._base_url}. Error: {exc}")

    def generate(self, question: str, context_chunks: list[dict]) -> str:
        context = self._build_context(context_chunks)
        prompt = self._build_prompt(question, context)
        payload = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": settings.temperature,
                "num_predict": settings.max_tokens,
            },
        }
        log.debug("Generating answer", model=self._model, context_chunks=len(context_chunks))
        try:
            response = requests.post(
                f"{self._base_url}/api/generate",
                json=payload,
                timeout=120,
            )
            response.raise_for_status()
            text = response.json()["response"].strip()
            log.debug("Generation done", chars=len(text))
            return text
        except requests.HTTPError as exc:
            log.error("Ollama request failed", status=exc.response.status_code, error=str(exc))
            raise
        except Exception as exc:
            log.error("Generation failed", error=str(exc))
            raise
