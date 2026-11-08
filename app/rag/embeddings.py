"""Embeddings with a *same-vector-space* fallback.

Switching embedding models mid-flight corrupts a vector index (different spaces / dimensions), so the
fallback for the HuggingFace Inference API is the **same model** executed locally with fastembed (ONNX).
Mistral embeddings have no safe fallback and simply retry.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Sequence

from langchain_core.embeddings import Embeddings

from app.config import Settings

log = logging.getLogger("rag.embeddings")

BATCH = 64


class FastEmbedLocal(Embeddings):
    """Local ONNX embeddings (lazy loaded -> instant API start-up)."""

    def __init__(self, model: str):
        self.model_name = model
        self._model = None
        self._failed_until = 0.0
        self._lock = threading.Lock()

    def _get(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    if time.monotonic() < self._failed_until:  # don't re-download for every request while offline
                        raise RuntimeError(f"local embedding model {self.model_name} could not be loaded (retrying later)")
                    from fastembed import TextEmbedding

                    log.info("loading local embedding model %s", self.model_name)
                    try:
                        self._model = TextEmbedding(model_name=self.model_name)
                    except Exception:
                        self._failed_until = time.monotonic() + 60
                        raise
        return self._model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self._get().embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


class ResilientEmbeddings(Embeddings):
    def __init__(self, primary: Embeddings, fallback: Embeddings | None, dim: int, retries: int = 2, name: str = ""):
        self.primary, self.fallback, self.dim, self.retries, self.name = primary, fallback, dim, retries, name
        self.using_fallback_until = 0.0

    def _call(self, fn_name: str, arg):
        last: Exception | None = None
        if time.monotonic() >= self.using_fallback_until:
            for attempt in range(self.retries + 1):
                try:
                    out = getattr(self.primary, fn_name)(arg)
                    return out
                except Exception as exc:  # noqa: BLE001
                    last = exc
                    log.warning("embedding primary failed (attempt %d): %s", attempt + 1, str(exc)[:200])
                    time.sleep(min(2**attempt * 0.5, 4))
        if self.fallback is not None:
            if last is not None:
                self.using_fallback_until = time.monotonic() + 60  # don't hammer a dead primary
            log.warning("using local embedding fallback")
            return getattr(self.fallback, fn_name)(arg)
        raise RuntimeError(f"Embedding provider failed: {last}")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), BATCH):
            out.extend(self._call("embed_documents", texts[i : i + BATCH]))
        if out and len(out[0]) != self.dim:
            raise ValueError(
                f"Embedding dimension {len(out[0])} != EMBEDDING_DIM={self.dim}. "
                "Fix EMBEDDING_DIM / EMBEDDING_MODEL (and re-create the Qdrant collection)."
            )
        return out

    def embed_query(self, text: str) -> list[float]:
        return self._call("embed_query", text)


def build_embeddings(s: Settings) -> Embeddings:
    provider = s.embedding_provider
    if provider == "local":
        return ResilientEmbeddings(FastEmbedLocal(s.embedding_model), None, s.embedding_dim, retries=0, name="local")
    if provider == "mistral":
        from langchain_mistralai import MistralAIEmbeddings

        key = s.secret("mistral_api_key")
        if not key:
            raise RuntimeError("EMBEDDING_PROVIDER=mistral requires MISTRAL_API_KEY")
        return ResilientEmbeddings(
            MistralAIEmbeddings(model=s.mistral_embedding_model, api_key=key), None, s.embedding_dim, name="mistral"
        )
    # huggingface (default)
    local = FastEmbedLocal(s.embedding_model)
    key = s.secret("huggingface_api_key")
    if not key:
        log.warning("HUGGINGFACE_API_KEY missing -> using local embeddings only")
        return ResilientEmbeddings(local, None, s.embedding_dim, retries=0, name="local")
    from langchain_huggingface import HuggingFaceEndpointEmbeddings

    hf = HuggingFaceEndpointEmbeddings(model=s.embedding_model, huggingfacehub_api_token=key)
    return ResilientEmbeddings(hf, local, s.embedding_dim, retries=1, name="huggingface")
