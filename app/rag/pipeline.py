"""The RAG orchestrator (planner -> cache -> retrieve -> rerank -> [web fallback] -> generate).

One streaming code path (``stream``) powers both the SSE endpoint and the blocking ``run`` helper, so behaviour
can never diverge between them. Every stage is timed and failures degrade gracefully:

    reranker down      -> vector order            cache down     -> skipped
    web search down    -> KB-only answer          primary LLM down -> next provider in the gateway chain
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import uuid
from typing import Any, AsyncIterator

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from app.config import Settings
from app.llm.gateway import AllProvidersFailed, LLMGateway
from app.llm.prompts import ANSWER, CANARY_VALUE, CHAT, PLANNER
from app.logging_setup import request_id_var
from app.rag.cache import SemanticCache
from app.rag.guardrails import REFUSAL, Guardrails
from app.rag.memory import ChatMemory
from app.rag.reranker import Reranker
from app.rag.vectorstore import KnowledgeBase
from app.rag.web_search import WebSearch
from app.schemas import ChatRequest, ChatResponse, Source

log = logging.getLogger("rag.pipeline")

_JSON_RE = re.compile(r"\{.*?\}", re.S)
_CITE_RE = re.compile(r"\[(\d{1,2})\]")
NO_LLM = "All language-model providers are currently unavailable. Please try again in a moment."


class PipelineError(RuntimeError):
    def __init__(self, message: str, status: int = 500, detail: list[str] | None = None):
        super().__init__(message)
        self.status, self.detail = status, detail or []


def _to_messages(history: list[dict]) -> list[BaseMessage]:
    return [HumanMessage(h["content"]) if h["role"] == "user" else AIMessage(h["content"]) for h in history]


def parse_plan(raw: str, fallback_query: str) -> dict[str, Any]:
    """Tolerant JSON extraction: models wrap JSON in prose/fences. Falls back to a safe technical plan."""
    plan = {"intent": "technical", "query": fallback_query, "standalone": False}
    m = _JSON_RE.search(raw or "")
    if not m:
        return plan
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return plan
    if str(data.get("intent", "")).lower().startswith("conv"):
        plan["intent"] = "conversational"
    q = str(data.get("query") or "").strip()
    if q:
        plan["query"] = q[:500]
    plan["standalone"] = bool(data.get("standalone", False))
    return plan


class RagPipeline:
    def __init__(self, s: Settings, gateway: LLMGateway, kb: KnowledgeBase, reranker: Reranker,
                 cache: SemanticCache, memory: ChatMemory, web: WebSearch, guard: Guardrails):
        self.s, self.llm, self.kb, self.reranker = s, gateway, kb, reranker
        self.cache, self.memory, self.web, self.guard = cache, memory, web, guard

    # ── helpers ──────────────────────────────────────────────────────────────────────────────────
    async def _plan(self, message: str, history: list[dict]) -> dict[str, Any]:
        words = len(message.split())
        if not history and words > 6:  # long first question: skip the router call (saves ~1 LLM round-trip)
            return {"intent": "technical", "query": message, "standalone": True}
        msgs = PLANNER.format_messages(domain=self.s.domain_description, history=_to_messages(history[-6:]), message=message)
        try:
            res = await self.llm.ainvoke(msgs)
            return parse_plan(res.text, message)
        except AllProvidersFailed:
            raise
        except Exception:  # noqa: BLE001
            return {"intent": "technical", "query": message, "standalone": not history}

    async def _retrieve(self, query: str, k: int, n: int) -> tuple[list[tuple[Document, float]], bool, dict[str, int]]:
        t = time.perf_counter()
        hits = await asyncio.to_thread(self.kb.search, query, k)
        t_ret = int((time.perf_counter() - t) * 1000)
        t = time.perf_counter()
        ranked, reranked = await asyncio.to_thread(self.reranker.rerank, query, hits, n)
        return ranked, reranked, {"retrieve": t_ret, "rerank": int((time.perf_counter() - t) * 1000)}

    def _is_grounded(self, ranked: list[tuple[Document, float]], reranked: bool) -> bool:
        if not ranked:
            return False
        top = max(score for _, score in ranked)
        return top >= (self.s.min_relevance_score if reranked else self.s.min_vector_score)

    def _build_context(self, ranked: list[tuple[Document, float]], web: list[dict]) -> tuple[str, list[Source]]:
        blocks, sources, used, n = [], [], 0, 0
        for doc, score in ranked:
            n += 1
            text = self.guard.sanitize_context(doc.page_content)
            if used + len(text) > self.s.max_context_chars:
                break
            used += len(text)
            m = doc.metadata
            page = m.get("page")
            label = f"{m.get('title', m.get('source', 'document'))}" + (f", p.{page}" if page else "")
            blocks.append(f"[{n}] ({label})\n{text}")
            sources.append(Source(id=n, title=m.get("title", "document"), origin="document", location=m.get("source", ""),
                                  page=page, score=round(float(score), 4), snippet=doc.page_content[:280]))
        for w in web:
            n += 1
            text = self.guard.sanitize_context(w["content"])[:1500]
            if used + len(text) > self.s.max_context_chars:
                break
            used += len(text)
            blocks.append(f"[{n}] (web: {w['title']})\n{text}")
            sources.append(Source(id=n, title=w["title"] or w["url"], origin="web", location=w["url"],
                                  score=round(float(w.get("score") or 0), 4), snippet=w["content"][:280]))
        return "\n\n".join(blocks) if blocks else "(no relevant context found)", sources

    @staticmethod
    def _mark_cited(answer: str, sources: list[Source]) -> None:
        cited = {int(x) for x in _CITE_RE.findall(answer)}
        for src in sources:
            src.cited = src.id in cited

    # ── main entry (streaming) ───────────────────────────────────────────────────────────────────
    async def stream(self, req: ChatRequest) -> AsyncIterator[dict[str, Any]]:
        t_start = time.perf_counter()
        rid = request_id_var.get()
        timings: dict[str, int] = {}
        thread_id = req.thread_id or uuid.uuid4().hex

        def final(answer: str, **kw: Any) -> dict[str, Any]:
            timings["total"] = int((time.perf_counter() - t_start) * 1000)
            resp = ChatResponse(thread_id=thread_id, answer=answer, latency_ms=timings, request_id=rid,
                                intent=kw.pop("intent", "technical"), standalone_query=kw.pop("standalone_query", req.message), **kw)
            return {"event": "final", "data": resp.model_dump()}

        # 1. input guardrails
        verdict = self.guard.check_input(req.message)
        if not verdict.allowed:
            log.info("input blocked: %s", verdict.reason)
            msg = REFUSAL if "injection" in verdict.reason else f"Request rejected: {verdict.reason}."
            yield {"event": "token", "data": {"text": msg}}
            yield final(msg, intent="blocked", blocked=True, grounded=False)
            return
        message = verdict.text

        history = await asyncio.to_thread(self.memory.history, thread_id, self.s.history_turns * 2)
        top_k, top_n = req.top_k or self.s.retrieval_top_k, req.top_n or self.s.rerank_top_n

        try:
            # 2. plan
            yield {"event": "status", "data": {"stage": "planning"}}
            t = time.perf_counter()
            plan = await self._plan(message, history)
            timings["plan"] = int((time.perf_counter() - t) * 1000)
            query = plan["query"]

            sources: list[Source] = []
            grounded, cached = True, False

            if plan["intent"] == "conversational":
                msgs = CHAT.format_messages(domain=self.s.domain_description, canary=CANARY_VALUE,
                                            history=_to_messages(history), message=message)
            else:
                # 3. semantic cache (only for self-contained questions)
                use_cache = self.s.semantic_cache_enabled if req.use_cache is None else req.use_cache
                if use_cache and plan["standalone"]:
                    t = time.perf_counter()
                    hit = await asyncio.to_thread(self.cache.get, query)
                    timings["cache"] = int((time.perf_counter() - t) * 1000)
                    if hit:
                        answer, _ = self.guard.check_output(hit["answer"])
                        yield {"event": "token", "data": {"text": answer}}
                        await self._remember(thread_id, message, answer, hit["sources"], "cache", None)
                        yield final(answer, intent="technical", standalone_query=query,
                                    sources=[Source(**x) for x in hit["sources"]], provider="semantic-cache",
                                    model=hit.get("model"), cached=True)
                        return

                # 4. retrieve + rerank
                yield {"event": "status", "data": {"stage": "retrieving", "query": query}}
                ranked, reranked, t_ret = await self._retrieve(query, top_k, top_n)
                timings.update(t_ret)
                grounded = self._is_grounded(ranked, reranked)

                # 5. web fallback
                web: list[dict] = []
                web_on = self.web.available if req.use_web_fallback is None else (req.use_web_fallback and bool(self.web.key))
                if not grounded and web_on:
                    yield {"event": "status", "data": {"stage": "web_search"}}
                    t = time.perf_counter()
                    web = await self.web.search(query)
                    timings["web"] = int((time.perf_counter() - t) * 1000)
                    if web:
                        grounded = True
                context, sources = self._build_context(ranked, web)
                web_note = ("5. Some context blocks come from the public web (marked 'web:'); say when you rely on them."
                            if web else "")
                msgs = ANSWER.format_messages(domain=self.s.domain_description, canary=CANARY_VALUE, web_note=web_note,
                                              history=_to_messages(history[-4:]), context=context, question=message)

            # 6. generate (streamed through the provider fallback chain)
            yield {"event": "status", "data": {"stage": "generating"}}
            t = time.perf_counter()
            parts: list[str] = []
            provider = model = None
            async for ev in self.llm.astream(msgs):
                if ev.type == "token":
                    parts.append(ev.text)
                    provider, model = ev.provider, ev.model
                    yield {"event": "token", "data": {"text": ev.text}}
                elif ev.type == "reset":
                    parts.clear()
                    yield {"event": "reset", "data": {"failed_provider": ev.provider}}
                elif ev.type == "done":
                    provider, model = ev.provider, ev.model
            timings["llm"] = int((time.perf_counter() - t) * 1000)

            answer, blocked = self.guard.check_output("".join(parts).strip())
            if blocked:
                yield {"event": "reset", "data": {"failed_provider": "guardrail"}}
                yield {"event": "token", "data": {"text": answer}}
            self._mark_cited(answer, sources)

            if (plan["intent"] == "technical" and plan["standalone"] and grounded and not blocked
                    and not any(x.origin == "web" for x in sources) and sources):
                use_cache = self.s.semantic_cache_enabled if req.use_cache is None else req.use_cache
                if use_cache:
                    await asyncio.to_thread(self.cache.put, query, answer, [x.model_dump() for x in sources], provider or "", model or "")

            await self._remember(thread_id, message, answer, [x.model_dump() for x in sources], provider, model)
            log.info("answered intent=%s provider=%s grounded=%s timings=%s", plan["intent"], provider, grounded, timings)
            yield final(answer, intent=plan["intent"], standalone_query=query, sources=sources, provider=provider,
                        model=model, grounded=grounded, blocked=blocked, cached=cached)
        except AllProvidersFailed as exc:
            log.error("%s", exc)
            yield {"event": "error", "data": {"message": NO_LLM, "status": 503, "detail": [f"{n}: {e}" for n, e in exc.errors]}}
        except Exception as exc:  # noqa: BLE001
            log.exception("pipeline failure")
            yield {"event": "error", "data": {"message": "Internal error while processing the request.", "status": 500, "detail": [type(exc).__name__]}}

    async def run(self, req: ChatRequest) -> ChatResponse:
        async for ev in self.stream(req):
            if ev["event"] == "final":
                return ChatResponse(**ev["data"])
            if ev["event"] == "error":
                raise PipelineError(ev["data"]["message"], ev["data"].get("status", 500), ev["data"].get("detail"))
        raise RuntimeError("pipeline ended without a result")

    async def _remember(self, thread_id: str, user: str, answer: str, sources: list[dict], provider: str | None, model: str | None) -> None:
        await asyncio.to_thread(self.memory.add, thread_id, "user", user)
        await asyncio.to_thread(self.memory.add, thread_id, "assistant", answer, {"sources": sources, "provider": provider, "model": model})
