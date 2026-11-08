"""LLM gateway: one interface over an ordered chain of providers.

Chain (only providers that have a key are enabled):
    1. OpenRouter   (primary key)
    2. OpenRouter   (backup key  - survives a rate-limited / exhausted primary key)
    3. Groq
    4. Mistral

Features
    * per-provider circuit breaker (skip a provider that keeps failing, probe it again after a cooldown)
    * fail-fast on auth errors (long cooldown) vs. transient errors (short cooldown)
    * streaming with *pre-first-token* fallback and *mid-stream* recovery (emits a ``reset`` event
      so the client can discard the partial answer before the next provider restarts it)
    * every call reports which provider/model actually answered
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage

from app.config import Settings

log = logging.getLogger("rag.gateway")

AUTH_COOLDOWN_S = 600.0


class AllProvidersFailed(RuntimeError):
    def __init__(self, errors: list[tuple[str, str]]):
        self.errors = errors
        detail = "; ".join(f"{n}: {e}" for n, e in errors) or "no providers configured"
        super().__init__(f"All LLM providers failed ({detail})")


def _text(content: Any) -> str:
    """Normalise LangChain message content (str | list[str|dict]) to plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for p in content:
        if isinstance(p, str):
            parts.append(p)
        elif isinstance(p, dict) and p.get("type") in (None, "text"):
            parts.append(p.get("text", ""))
    return "".join(parts)


def _is_auth_error(exc: BaseException) -> bool:
    s = f"{type(exc).__name__} {exc}".lower()
    return any(t in s for t in ("401", "403", "invalid api key", "invalid_api_key", "authentication", "unauthorized"))


@dataclass
class Breaker:
    threshold: int
    cooldown_s: float
    failures: int = 0
    open_until: float = 0.0
    last_error: str = ""

    def available(self) -> bool:
        return time.monotonic() >= self.open_until

    def success(self) -> None:
        self.failures = 0
        self.open_until = 0.0

    def failure(self, err: str, fatal: bool = False) -> None:
        self.failures += 1
        self.last_error = err[:300]
        if fatal:
            self.open_until = time.monotonic() + AUTH_COOLDOWN_S
        elif self.failures >= self.threshold:
            self.open_until = time.monotonic() + self.cooldown_s


@dataclass
class Provider:
    name: str
    model: str
    llm: BaseChatModel
    breaker: Breaker
    calls: int = 0
    errors: int = 0


@dataclass
class GatewayResult:
    text: str
    provider: str
    model: str
    attempts: list[tuple[str, str]] = field(default_factory=list)  # (provider, error) for failed hops


@dataclass
class StreamEvent:
    type: str  # "token" | "reset" | "done"
    text: str = ""
    provider: str = ""
    model: str = ""


# ── provider construction (lazy imports keep start-up fast and optional deps optional) ──────────
def _build_openrouter(key: str, model: str, s: Settings) -> BaseChatModel:
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=model,
        api_key=key,
        base_url=s.openrouter_base_url,
        temperature=s.llm_temperature,
        max_tokens=s.llm_max_tokens,
        timeout=s.llm_timeout_s,
        max_retries=s.llm_max_retries,
        default_headers={"HTTP-Referer": "https://localhost/enterprise-rag", "X-Title": "Enterprise RAG"},
    )


def _build_groq(key: str, model: str, s: Settings) -> BaseChatModel:
    from langchain_groq import ChatGroq

    return ChatGroq(
        model=model,
        api_key=key,
        temperature=s.llm_temperature,
        max_tokens=s.llm_max_tokens,
        timeout=s.llm_timeout_s,
        max_retries=s.llm_max_retries,
    )


def _build_mistral(key: str, model: str, s: Settings) -> BaseChatModel:
    from langchain_mistralai import ChatMistralAI

    return ChatMistralAI(
        model=model,
        api_key=key,
        temperature=s.llm_temperature,
        max_tokens=s.llm_max_tokens,
        timeout=int(s.llm_timeout_s),
        max_retries=s.llm_max_retries,
    )


def build_providers(s: Settings) -> list[Provider]:
    specs: list[tuple[str, str, str | None, Callable[[str, str, Settings], BaseChatModel]]] = [
        ("openrouter", s.openrouter_model, s.secret("openrouter_api_key"), _build_openrouter),
        (
            "openrouter-backup",
            s.openrouter_fallback_model or s.openrouter_model,
            s.secret("openrouter_fallback_api_key"),
            _build_openrouter,
        ),
        ("groq", s.groq_model, s.secret("groq_api_key"), _build_groq),
        ("mistral", s.mistral_model, s.secret("mistral_api_key"), _build_mistral),
    ]
    providers: list[Provider] = []
    for name, model, key, builder in specs:
        if not key:
            log.info("provider %s disabled (no API key)", name)
            continue
        providers.append(
            Provider(name, model, builder(key, model, s), Breaker(s.breaker_failure_threshold, s.breaker_cooldown_s))
        )
    return providers


class LLMGateway:
    def __init__(self, providers: Sequence[Provider], timeout_s: float = 45.0):
        self.providers = list(providers)
        self.timeout_s = timeout_s

    @classmethod
    def from_settings(cls, s: Settings) -> "LLMGateway":
        gw = cls(build_providers(s), timeout_s=s.llm_timeout_s)
        log.info("LLM chain: %s", " -> ".join(f"{p.name}({p.model})" for p in gw.providers) or "NONE")
        return gw

    # ── ordering: healthy providers first (in configured order); tripped ones are a last resort ──
    def _order(self) -> list[Provider]:
        healthy = [p for p in self.providers if p.breaker.available()]
        tripped = [p for p in self.providers if not p.breaker.available()]
        return healthy + tripped if not healthy else healthy

    def _record_failure(self, p: Provider, exc: BaseException) -> str:
        err = f"{type(exc).__name__}: {str(exc)[:200]}"
        p.errors += 1
        p.breaker.failure(err, fatal=_is_auth_error(exc))
        log.warning("provider %s failed (%s) -> falling back", p.name, err)
        return err

    # ── non-streaming ───────────────────────────────────────────────────────────────────────────
    async def ainvoke(self, messages: list[BaseMessage]) -> GatewayResult:
        attempts: list[tuple[str, str]] = []
        for p in self._order():
            p.calls += 1
            try:
                msg = await asyncio.wait_for(p.llm.ainvoke(messages), timeout=self.timeout_s + 5)
                text = _text(msg.content).strip()
                if not text:
                    raise ValueError("empty completion")
                p.breaker.success()
                return GatewayResult(text, p.name, p.model, attempts)
            except Exception as exc:  # noqa: BLE001 - any provider failure triggers fallback
                attempts.append((p.name, self._record_failure(p, exc)))
        raise AllProvidersFailed(attempts)

    # ── streaming ───────────────────────────────────────────────────────────────────────────────
    async def astream(self, messages: list[BaseMessage]) -> AsyncIterator[StreamEvent]:
        attempts: list[tuple[str, str]] = []
        for p in self._order():
            p.calls += 1
            started = False
            it = p.llm.astream(messages).__aiter__()
            try:
                while True:
                    try:
                        chunk = await asyncio.wait_for(it.__anext__(), timeout=self.timeout_s)
                    except StopAsyncIteration:
                        break
                    piece = _text(chunk.content)
                    if piece:
                        started = True
                        yield StreamEvent("token", piece, p.name, p.model)
                if not started:
                    raise ValueError("empty completion")
                p.breaker.success()
                yield StreamEvent("done", "", p.name, p.model)
                return
            except Exception as exc:  # noqa: BLE001
                attempts.append((p.name, self._record_failure(p, exc)))
                if started:  # client already rendered partial text -> tell it to discard
                    yield StreamEvent("reset", "", p.name, p.model)
        raise AllProvidersFailed(attempts)

    # ── introspection ───────────────────────────────────────────────────────────────────────────
    def status(self) -> list[dict[str, Any]]:
        now = time.monotonic()
        return [
            {
                "provider": p.name,
                "model": p.model,
                "state": "open" if not p.breaker.available() else "closed",
                "retry_in_s": max(0, round(p.breaker.open_until - now)) if not p.breaker.available() else 0,
                "consecutive_failures": p.breaker.failures,
                "calls": p.calls,
                "errors": p.errors,
                "last_error": p.breaker.last_error,
            }
            for p in self.providers
        ]
