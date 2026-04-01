import structlog
from typing import List, Optional
from qdrant_client import QdrantClient, models
from qdrant_client.hybrid.fusion import reciprocal_rank_fusion

from src.config import settings

log = structlog.get_logger(__name__)


class VectorStore:
    def __init__(self) -> None:
        self._client = QdrantClient(location=settings.qdrant_location)
        self._collection = settings.qdrant_collection

        log.info("Loading embedding model", model=settings.embed_model)
        self._client.set_model(settings.embed_model)
        self._client.set_sparse_model(settings.sparse_embed_model)

        # Keep direct references to fastembed models for use in search()
        # (the deprecated client.query() is broken in qdrant-client 1.17.x)
        embedder = self._client._model_embedder.embedder
        self._dense_model = embedder.get_or_init_model(
            model_name=settings.embed_model, deprecated=True
        )
        self._sparse_model = embedder.get_or_init_sparse_model(
            model_name=settings.sparse_embed_model, deprecated=True
        )

        self._reranker = None
        if settings.enable_reranking:
            from fastembed.rerank.cross_encoder import TextCrossEncoder

            log.info("Loading reranker", model=settings.rerank_model)
            self._reranker = TextCrossEncoder(model_name=settings.rerank_model)

    def add_chunks(self, chunks: List[dict], batch_size: int = 64) -> int:
        if not chunks:
            return 0

        texts = [chunk["text"] for chunk in chunks]
        metadatas = [chunk.get("metadata", {}) for chunk in chunks]

        if not self._client.collection_exists(self._collection):
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=self._client.get_fastembed_vector_params(),
                sparse_vectors_config=self._client.get_fastembed_sparse_vector_params(),
            )

        self._client.add(
            collection_name=self._collection,
            documents=texts,
            metadata=metadatas,
            batch_size=batch_size,
        )

        log.info("Indexing complete", chunks=len(texts), collection=self._collection)
        return len(texts)

    def search(self, query: str, top_k: Optional[int] = None) -> List[dict]:
        if not self._client.collection_exists(self._collection):
            return []

        k = top_k or settings.top_k
        fetch_limit = max(k * 4, settings.rerank_candidates) if self._reranker else k

        dense_vector = list(self._dense_model.query_embed(query))[0].tolist()

        sparse_embed = list(self._sparse_model.query_embed(query))[0]
        sparse_vector = models.SparseVector(
            indices=sparse_embed.indices.tolist(),
            values=sparse_embed.values.tolist(),
        )

        dense_field = self._client.get_vector_field_name()
        sparse_field = self._client.get_sparse_vector_field_name()

        dense_results = self._client.query_points(
            collection_name=self._collection,
            query=dense_vector,
            using=dense_field,
            limit=fetch_limit,
            with_payload=True,
        ).points
        sparse_results = self._client.query_points(
            collection_name=self._collection,
            query=sparse_vector,
            using=sparse_field,
            limit=fetch_limit,
            with_payload=True,
        ).points

        fused = reciprocal_rank_fusion([dense_results, sparse_results], limit=fetch_limit)

        hits = [
            {"text": p.payload.get("document", ""), "metadata": p.payload, "score": round(p.score, 4)}
            for p in fused
        ]

        if self._reranker and len(hits) > 1:
            texts = [hit["text"] for hit in hits]
            scores = list(self._reranker.rerank(query, texts))
            # Cross-encoder scores are logits and can be negative — do not threshold them
            hits = sorted(
                [{**hit, "rerank_score": round(score, 4)} for score, hit in zip(scores, hits)],
                key=lambda h: h["rerank_score"],
                reverse=True,
            )
        else:
            hits = [h for h in hits if h["score"] >= settings.score_threshold]

        return hits[:k]

    def count(self) -> int:
        try:
            if self._client.collection_exists(self._collection):
                return self._client.count(collection_name=self._collection).count
            return 0
        except Exception:
            return 0

    def reset(self) -> None:
        if self._client.collection_exists(self._collection):
            self._client.delete_collection(self._collection)
            log.info("Collection deleted", collection=self._collection)
