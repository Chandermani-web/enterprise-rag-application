from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from app.config import Settings
from app.container import Container
from app.llm.gateway import Breaker, LLMGateway, Provider
from app.rag.cache import SemanticCache
from app.rag.guardrails import Guardrails
from app.rag.ingestion import Ingestor
from app.rag.memory import ChatMemory
from app.rag.pipeline import RagPipeline
from app.rag.reranker import Reranker
from app.rag.vectorstore import KnowledgeBase
from app.rag.web_search import WebSearch
from qdrant_client import QdrantClient

from tests.fakes import HashEmbeddings, ScriptedChat, smart_responder


def make_settings(tmp_path: Path, **over) -> Settings:
    base = dict(_env_file=None, qdrant_in_memory=True, state_dir=tmp_path / "state", data_dir=tmp_path / "data",
                embedding_dim=64, rerank_enabled=False, rate_limit_per_minute=1000, min_vector_score=0.05,
                semantic_cache_threshold=0.9)
    base.update(over)
    s = Settings(**base)
    s.state_dir.mkdir(parents=True, exist_ok=True)
    s.data_dir.mkdir(parents=True, exist_ok=True)
    return s


def provider(name: str, **kw) -> Provider:
    return Provider(name, f"{name}-model", ScriptedChat(responder=kw.pop("responder", smart_responder), **kw), Breaker(3, 60))


@pytest.fixture
def settings(tmp_path):
    return make_settings(tmp_path)


@pytest.fixture
def docs_dir(settings):
    d = settings.data_dir
    (d / "kubernetes.md").write_text(textwrap.dedent("""
        # SR-IOV on Kubernetes
        To configure Intel SR-IOV on Kubernetes install the SR-IOV network device plugin and create a
        SriovNetworkNodePolicy that selects the physical function. Then request the virtual function in the pod spec.

        # Horizontal Pod Autoscaler
        The Horizontal Pod Autoscaler (HPA) scales the number of pod replicas based on observed CPU utilisation.
        Configure it with kubectl autoscale deployment and target a CPU percentage.
        """))
    (d / "redis.txt").write_text("Redis semantic caching stores embeddings of previous questions so that similar "
                                 "questions are answered instantly from cache without calling the language model.")
    return d


def build(settings: Settings, providers: list[Provider], web: WebSearch | None = None) -> Container:
    emb = HashEmbeddings(64)
    client = QdrantClient(":memory:")
    kb = KnowledgeBase(settings, client, emb)
    cache = SemanticCache(settings, client, emb)
    memory = ChatMemory(settings.state_dir / "m.sqlite3")
    reranker = Reranker(settings)
    gw = LLMGateway(providers, timeout_s=5)
    pipe = RagPipeline(settings, gw, kb, reranker, cache, memory, web or WebSearch(settings), Guardrails(True, 4000))
    return Container(settings, gw, kb, cache, Ingestor(settings, kb), memory, reranker, pipe)


@pytest.fixture
def container(settings, docs_dir):
    c = build(settings, [provider("openrouter")])
    c.ingestor.run(docs_dir)
    return c
