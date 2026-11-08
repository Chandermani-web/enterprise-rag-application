"""Qdrant access: collection bootstrap, LangChain vector store, document management helpers."""
from __future__ import annotations

import logging
import warnings
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models

from app.config import Settings

log = logging.getLogger("rag.qdrant")
warnings.filterwarnings("ignore", message="Payload indexes have no effect in the local Qdrant")

SOURCE_KEY = "metadata.source"


def build_client(s: Settings) -> QdrantClient:
    if s.qdrant_in_memory or not s.qdrant_url:
        if not s.qdrant_in_memory:
            log.warning("QDRANT_CLUSTER_ENDPOINT not set -> using IN-MEMORY Qdrant (data is lost on restart)")
        return QdrantClient(":memory:")
    return QdrantClient(url=s.qdrant_url, api_key=s.secret("qdrant_api_key"), timeout=30, prefer_grpc=False)


def ensure_collection(client: QdrantClient, name: str, dim: int, keyword_indexes: tuple[str, ...] = (),
                      float_indexes: tuple[str, ...] = ()) -> None:
    if client.collection_exists(name):
        info = client.get_collection(name)
        vectors = info.config.params.vectors
        existing = vectors.size if isinstance(vectors, models.VectorParams) else None
        if existing is not None and existing != dim:
            raise RuntimeError(
                f"Qdrant collection '{name}' has dim={existing} but EMBEDDING_DIM={dim}. "
                "Use a new QDRANT_COLLECTION or delete the old one."
            )
    else:
        client.create_collection(name, vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE))
        log.info("created Qdrant collection %s (dim=%d)", name, dim)
    wanted = [(f, models.PayloadSchemaType.KEYWORD) for f in keyword_indexes] + [
        (f, models.PayloadSchemaType.FLOAT) for f in float_indexes
    ]
    for field, schema in wanted:
        try:
            client.create_payload_index(name, field_name=field, field_schema=schema)
        except Exception:  # noqa: BLE001 - already exists / not supported in local mode
            pass


class KnowledgeBase:
    """Thin facade over the Qdrant collection used for documents."""

    def __init__(self, s: Settings, client: QdrantClient, embeddings: Embeddings):
        self.s, self.client, self.embeddings = s, client, embeddings
        self.collection = s.qdrant_collection
        ensure_collection(client, self.collection, s.embedding_dim, (SOURCE_KEY, "metadata.content_hash"))
        self.store = QdrantVectorStore(
            client=client, collection_name=self.collection, embedding=embeddings, validate_collection_config=False
        )

    # writes
    def add(self, docs: list[Document], ids: list[str]) -> int:
        self.store.add_documents(docs, ids=ids)
        return len(docs)

    def delete_source(self, source: str) -> None:
        self.client.delete(
            self.collection,
            points_selector=models.FilterSelector(filter=self._source_filter(source)),
            wait=True,
        )

    # reads
    def search(self, query: str, k: int, score_threshold: float | None = None) -> list[tuple[Document, float]]:
        return self.store.similarity_search_with_score(query, k=k, score_threshold=score_threshold)

    def source_hash(self, source: str) -> str | None:
        points, _ = self.client.scroll(
            self.collection, scroll_filter=self._source_filter(source), limit=1, with_payload=["metadata.content_hash"]
        )
        if not points:
            return None
        return (points[0].payload or {}).get("metadata", {}).get("content_hash")

    def list_documents(self) -> list[dict[str, Any]]:
        counts: dict[str, int] = {}
        offset = None
        while True:
            points, offset = self.client.scroll(
                self.collection, limit=512, offset=offset, with_payload=["metadata.source"], with_vectors=False
            )
            for p in points:
                src = (p.payload or {}).get("metadata", {}).get("source", "unknown")
                counts[src] = counts.get(src, 0) + 1
            if offset is None:
                break
        return [{"source": k, "chunks": v} for k, v in sorted(counts.items())]

    def count(self) -> int:
        return self.client.count(self.collection, exact=True).count

    def healthy(self) -> bool:
        try:
            self.client.get_collections()
            return True
        except Exception:  # noqa: BLE001
            return False

    @staticmethod
    def _source_filter(source: str) -> models.Filter:
        return models.Filter(must=[models.FieldCondition(key=SOURCE_KEY, match=models.MatchValue(value=source))])
