import argparse
import sys
import warnings
import structlog

warnings.filterwarnings("ignore", category=DeprecationWarning)

log = structlog.get_logger(__name__)

from src.benchmark.samples import TEST_SAMPLES
from src.benchmark.utils import build_llm, build_embeddings, build_metrics

ALL_METRICS = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
]


def run_pipeline(samples: list) -> list:
    from src.ingestion.embedder import VectorStore
    from src.agent import QAAgent
    from src.ingestion.loaders import load_directory

    log.info("--> Initializing In-Memory DB and loading docs...")
    store = VectorStore()
    chunks = load_directory("./data")
    if not chunks:
        log.info("Error: No docs found in ./data")
        sys.exit(1)
    store.add_chunks(chunks)

    agent = QAAgent(store)
    enriched = []

    log.info(f"--> Running RAG on {len(samples)} samples...")
    for s in samples:
        question = s["user_input"]
        retrieved = store.search(question)
        response = agent.ask(question).answer
        enriched.append(
            {
                **s,
                "retrieved_contexts": [c["text"] for c in retrieved],
                "response": response,
            }
        )
    return enriched


def main():
    parser = argparse.ArgumentParser(description="RAGAS benchmark for the RAG pipeline")
    parser.add_argument("--provider", choices=["openai", "ollama"], help="LLM provider for evaluation judge")
    parser.add_argument("--metrics", nargs="+", choices=ALL_METRICS, default=ALL_METRICS)
    parser.add_argument("--output", help="Save per-sample results to JSON file")
    args = parser.parse_args()

    from src.config import settings
    from ragas import EvaluationDataset, SingleTurnSample, evaluate

    provider = args.provider or settings.llm_provider
    log.info(f"=== RAGAS BENCHMARK (provider: {provider}) ===")

    enriched = run_pipeline(TEST_SAMPLES)

    ragas_samples = [SingleTurnSample(**s) for s in enriched]
    dataset = EvaluationDataset(ragas_samples)

    log.info("--> Evaluating with judge models...")
    llm = build_llm(provider)
    embeddings = build_embeddings()
    metrics = build_metrics(args.metrics, llm, embeddings)

    result = evaluate(dataset=dataset, metrics=metrics)

    df = result.to_pandas()

    log.info("\n" + "=" * 40)
    log.info("AVERAGE SCORES:")
    for metric in args.metrics:
        if metric in df.columns:
            log.info(f"  {metric:20}: {df[metric].mean():.4f}")
    log.info("=" * 40)

    log.info("\nPER-SAMPLE RESULTS:")
    for i, row in df.iterrows():
        log.info(f"[{i + 1}] Q: {row['user_input'][:60]}...")
        for m in args.metrics:
            if m in row and row[m] == row[m]:  # skip NaN
                log.info(f"      {m:20}: {row[m]:.4f}")

    if args.output:
        df.to_json(args.output, orient="records", force_ascii=False, indent=2)
        log.debug(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
