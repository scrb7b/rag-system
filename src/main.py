import io
import structlog
import sys
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="qdrant_client")

from src.ingestion.loaders import load_directory
from src.ingestion.embedder import VectorStore
from src.generation import get_llm
from src.agent import QAAgent

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", line_buffering=True)
sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding="utf-8")

log = structlog.get_logger(__name__)


def run_in_memory_pipeline(data_dir: str):
    log.info("Initializing vector store...")
    store = VectorStore()

    log.info(f"Loading documents from: {data_dir}")
    chunks = load_directory(data_dir)

    if not chunks:
        log.error("No documents found. Check the ./data folder.")
        return

    log.info(f"Indexing {len(chunks)} chunks...")
    store.add_chunks(chunks)

    llm = get_llm()
    agent = QAAgent(store, llm)

    print("\nChat is ready. Type your question or 'exit' to quit.\n", flush=True)

    while True:
        try:
            question = input("Question: ").strip()
            if not question or question.lower() in ("exit", "quit", "q"):
                print("Bye.")
                break

            result = agent.ask(question)
            print(f"\n{result.answer}\n", flush=True)

            if result.chunks_used:
                score_key = "rerank_score" if "rerank_score" in result.chunks_used[0] else "score"
                sources = [
                    f"{chunk.get('metadata', {}).get('filename', 'unknown')} ({chunk.get(score_key, 0):.2f})"
                    for chunk in result.chunks_used
                ]
                print(f"Sources: {', '.join(sources)}\n", flush=True)

        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        except UnicodeDecodeError:
            print("Input encoding error. Make sure your terminal uses UTF-8.", flush=True)


if __name__ == "__main__":
    data_path = "./data"
    run_in_memory_pipeline(data_path)
