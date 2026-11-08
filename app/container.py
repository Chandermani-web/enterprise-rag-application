"""Dependency container: builds every service once, in the right order (lazy where it matters)."""
from __future__ import annotations

from dataclasses import dataclass

from app.config import Settings
from app.llm.gateway import LLMGateway
from app.rag.cache import SemanticCache
from app.rag.embeddings import build_embeddings
from app.rag.guardrails import Guardrails
from app.rag.ingestion import Ingestor
from app.rag.memory import ChatMemory
from app.rag.pipeline import RagPipeline
from app.rag.reranker import Reranker
from app.rag.vectorstore import KnowledgeBase, build_client
from app.rag.web_search import WebSearch


@dataclass
class Container:
    settings: Settings
    gateway: LLMGateway
    kb: KnowledgeBase
    cache: SemanticCache
    ingestor: Ingestor
    memory: ChatMemory
    reranker: Reranker
    pipeline: RagPipeline


def build_container(s: Settings) -> Container:
    client = build_client(s)
    embeddings = build_embeddings(s)
    kb = KnowledgeBase(s, client, embeddings)
    cache = SemanticCache(s, client, embeddings)
    memory = ChatMemory(s.state_dir / "memory.sqlite3")
    reranker = Reranker(s)
    gateway = LLMGateway.from_settings(s)
    guard = Guardrails(s.guardrails_enabled, s.max_input_chars, s.redact_pii)
    pipeline = RagPipeline(s, gateway, kb, reranker, cache, memory, WebSearch(s), guard)
    return Container(s, gateway, kb, cache, Ingestor(s, kb), memory, reranker, pipeline)
