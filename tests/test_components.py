import pytest
from langchain_core.documents import Document
from qdrant_client import QdrantClient

from app.rag.embeddings import ResilientEmbeddings
from app.rag.reranker import Reranker
from app.rag.vectorstore import ensure_collection
from tests.conftest import make_settings
from tests.fakes import HashEmbeddings


class Broken(HashEmbeddings):
    def embed_documents(self, texts):
        raise ConnectionError("hf down")

    def embed_query(self, text):
        raise ConnectionError("hf down")


def test_embedding_falls_back_to_local_and_stays_there(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda *_: None)
    primary, local = Broken(64), HashEmbeddings(64)
    e = ResilientEmbeddings(primary, local, 64, retries=1)
    assert len(e.embed_query("hello")) == 64 and local.calls == 1
    e.embed_documents(["a", "b"])  # primary is skipped during cool-down (no hammering)
    assert local.calls == 2


def test_embedding_without_fallback_raises(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda *_: None)
    with pytest.raises(RuntimeError, match="Embedding provider failed"):
        ResilientEmbeddings(Broken(64), None, 64, retries=0).embed_query("x")


def test_embedding_dimension_mismatch_is_caught():
    with pytest.raises(ValueError, match="EMBEDDING_DIM"):
        ResilientEmbeddings(HashEmbeddings(32), None, 64).embed_documents(["x"])


def test_collection_dim_mismatch_guard():
    c = QdrantClient(":memory:")
    ensure_collection(c, "col", 64)
    ensure_collection(c, "col", 64)  # idempotent
    with pytest.raises(RuntimeError, match="dim=64"):
        ensure_collection(c, "col", 384)


class FakeRanker:
    def rerank(self, req):  # reverse order, scores descending
        n = len(req.passages)
        return [{"id": p["id"], "score": (p["id"] + 1) / n} for p in sorted(req.passages, key=lambda p: -p["id"])]


def hits(n=4):
    return [(Document(page_content=f"doc {i}", metadata={"i": i}), 0.5) for i in range(n)]


def test_reranker_reorders_and_trims(tmp_path):
    r = Reranker(make_settings(tmp_path, rerank_enabled=True))
    r._ranker = FakeRanker()
    out, did = r.rerank("q", hits(), 2)
    assert did and [d.metadata["i"] for d, _ in out] == [3, 2]


def test_reranker_unavailable_falls_back_to_vector_order(tmp_path):
    r = Reranker(make_settings(tmp_path, rerank_enabled=True))
    r._failed = True  # model could not be loaded
    out, did = r.rerank("q", hits(), 2)
    assert not did and [d.metadata["i"] for d, _ in out] == [0, 1]


def test_reranker_runtime_error_falls_back(tmp_path):
    class Boom:
        def rerank(self, req):
            raise MemoryError("oom")

    r = Reranker(make_settings(tmp_path, rerank_enabled=True))
    r._ranker = Boom()
    out, did = r.rerank("q", hits(), 3)
    assert not did and len(out) == 3
