"""Semantic answer cache stored in a dedicated Qdrant collection (no extra infrastructure needed).

A new question is embedded; if a cached question is >= threshold cosine-similar and not expired, the stored
answer is returned without touching the LLM. The cache is flushed whenever documents are (re)ingested so a
stale answer can never outlive the knowledge it was built from.
"""
from __future__ import annotations

import json
import logging
import time
import uuid

from langchain_core.embeddings import Embeddings
from qdrant_client import QdrantClient, models

from app.config import Settings
from app.rag.vectorstore import ensure_collection

log = logging.getLogger("rag.cache")


class SemanticCache:
    def __init__(self, s: Settings, client: QdrantClient, embeddings: Embeddings):
        self.s, self.client, self.emb = s, client, embeddings
        self.collection = s.qdrant_cache_collection
        self.enabled = s.semantic_cache_enabled
        self.hits = self.misses = 0
        if self.enabled:
            ensure_collection(client, self.collection, s.embedding_dim, float_indexes=("expires_at",))

    def get(self, query: str) -> dict | None:
        if not self.enabled:
            return None
        try:
            vec = self.emb.embed_query(query)
            res = self.client.query_points(
                self.collection,
                query=vec,
                limit=1,
                score_threshold=self.s.semantic_cache_threshold,
                query_filter=models.Filter(
                    must=[models.FieldCondition(key="expires_at", range=models.Range(gt=time.time()))]
                ),
                with_payload=True,
            )
            if res.points:
                self.hits += 1
                p = res.points[0].payload or {}
                return {"answer": p["answer"], "sources": json.loads(p.get("sources", "[]")), "provider": p.get("provider"),
                        "model": p.get("model"), "similarity": res.points[0].score}
            self.misses += 1
        except Exception as exc:  # noqa: BLE001 - the cache must never break a request
            log.warning("cache get failed: %s", str(exc)[:200])
        return None

    def put(self, query: str, answer: str, sources: list[dict], provider: str, model: str) -> None:
        if not self.enabled:
            return
        try:
            now = time.time()
            self.client.upsert(
                self.collection,
                points=[
                    models.PointStruct(
                        id=str(uuid.uuid4()),
                        vector=self.emb.embed_query(query),
                        payload={
                            "query": query, "answer": answer, "sources": json.dumps(sources), "provider": provider,
                            "model": model, "created_at": now, "expires_at": now + self.s.semantic_cache_ttl_s,
                        },
                    )
                ],
            )
            self.client.delete(
                self.collection,
                points_selector=models.FilterSelector(
                    filter=models.Filter(must=[models.FieldCondition(key="expires_at", range=models.Range(lt=now))])
                ),
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("cache put failed: %s", str(exc)[:200])

    def clear(self) -> None:
        if not self.enabled:
            return
        try:
            self.client.delete_collection(self.collection)
            ensure_collection(self.client, self.collection, self.s.embedding_dim, float_indexes=("expires_at",))
        except Exception as exc:  # noqa: BLE001
            log.warning("cache clear failed: %s", str(exc)[:200])
