"""Deterministic fakes so the whole stack is testable offline."""
from __future__ import annotations

import hashlib
import math
import re
from typing import Any, AsyncIterator, Callable, Iterator

from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult


class HashEmbeddings(Embeddings):
    """Bag-of-words hashing embeddings: similar wording -> high cosine similarity."""

    def __init__(self, dim: int = 64):
        self.dim = dim
        self.calls = 0

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for w in re.findall(r"[a-z0-9]+", text.lower()):
            h = int(hashlib.md5(w.encode()).hexdigest(), 16)
            v[h % self.dim] += 1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        self.calls += 1
        return self._vec(text)


class ScriptedChat(BaseChatModel):
    """Chat model driven by a python function. Can fail up front or after N streamed chunks."""

    responder: Any = None
    fail: str | None = None  # raise this message immediately
    fail_after_chunks: int | None = None  # stream N chunks, then raise
    calls: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _text(self, messages: list[BaseMessage]) -> str:
        return self.responder(messages) if self.responder else "ok"

    def _generate(self, messages, stop=None, run_manager: CallbackManagerForLLMRun | None = None, **kw) -> ChatResult:
        self.calls += 1
        if self.fail:
            raise RuntimeError(self.fail)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=self._text(messages)))])

    async def _astream(self, messages, stop=None, run_manager: AsyncCallbackManagerForLLMRun | None = None, **kw) -> AsyncIterator[ChatGenerationChunk]:
        self.calls += 1
        if self.fail:
            raise RuntimeError(self.fail)
        words = self._text(messages).split(" ")
        for i, w in enumerate(words):
            if self.fail_after_chunks is not None and i >= self.fail_after_chunks:
                raise RuntimeError("stream broke")
            yield ChatGenerationChunk(message=AIMessageChunk(content=w + (" " if i < len(words) - 1 else "")))


def smart_responder(messages: list[BaseMessage]) -> str:
    """Behaves like planner / answerer / chatter depending on the system prompt."""
    system = messages[0].content if messages else ""
    last = messages[-1].content
    if "routing step" in system:
        msg = last.replace("Latest user message: ", "")
        if len(msg.split()) <= 3 and re.search(r"\b(hi|hello|thanks|hey)\b", msg.lower()):
            return '{"intent": "conversational", "query": "", "standalone": true}'
        return '```json\n{"intent": "technical", "query": "%s", "standalone": true}\n```' % msg.replace('"', "")
    if "CONTEXT" in last:
        first_ctx = last.split("CONTEXT:\n", 1)[1][:160].replace("\n", " ")
        return f"According to the docs [1]: {first_ctx}"
    return "Hello! How can I help you with the documentation today?"
