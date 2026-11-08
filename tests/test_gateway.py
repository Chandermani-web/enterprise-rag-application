import pytest
from langchain_core.messages import HumanMessage

from app.llm.gateway import AllProvidersFailed, build_providers
from tests.conftest import build, make_settings, provider


async def collect(gw, msgs):
    return [e async for e in gw.astream(msgs)]


@pytest.mark.asyncio
async def test_fallback_chain_order(tmp_path):
    c = build(make_settings(tmp_path), [
        provider("openrouter", fail="429 rate limit"),
        provider("openrouter-backup", fail="500 boom"),
        provider("groq", responder=lambda m: "from groq"),
        provider("mistral", responder=lambda m: "from mistral"),
    ])
    res = await c.gateway.ainvoke([HumanMessage("hi")])
    assert (res.provider, res.text) == ("groq", "from groq")
    assert [a[0] for a in res.attempts] == ["openrouter", "openrouter-backup"]


@pytest.mark.asyncio
async def test_all_fail_raises_with_details(tmp_path):
    c = build(make_settings(tmp_path), [provider("a", fail="x"), provider("b", fail="y")])
    with pytest.raises(AllProvidersFailed) as e:
        await c.gateway.ainvoke([HumanMessage("hi")])
    assert len(e.value.errors) == 2


@pytest.mark.asyncio
async def test_circuit_breaker_opens_and_skips(tmp_path):
    bad = provider("bad", fail="503")
    good = provider("good", responder=lambda m: "fine")
    c = build(make_settings(tmp_path), [bad, good])
    for _ in range(3):
        await c.gateway.ainvoke([HumanMessage("hi")])
    assert bad.breaker.available() is False
    calls_before = bad.llm.calls
    await c.gateway.ainvoke([HumanMessage("hi")])
    assert bad.llm.calls == calls_before  # skipped while open
    assert c.gateway.status()[0]["state"] == "open"


@pytest.mark.asyncio
async def test_auth_error_trips_breaker_immediately(tmp_path):
    bad = provider("bad", fail="401 Unauthorized invalid api key")
    c = build(make_settings(tmp_path), [bad, provider("good", responder=lambda m: "ok")])
    await c.gateway.ainvoke([HumanMessage("hi")])
    assert bad.breaker.available() is False


@pytest.mark.asyncio
async def test_stream_prefirst_token_fallback(tmp_path):
    c = build(make_settings(tmp_path), [provider("a", fail="down"), provider("b", responder=lambda m: "one two three")])
    evs = await collect(c.gateway, [HumanMessage("x")])
    assert "".join(e.text for e in evs if e.type == "token") == "one two three"
    assert not any(e.type == "reset" for e in evs)
    assert evs[-1].type == "done" and evs[-1].provider == "b"


@pytest.mark.asyncio
async def test_stream_midstream_failure_emits_reset_then_restarts(tmp_path):
    c = build(make_settings(tmp_path), [
        provider("a", responder=lambda m: "alpha beta gamma delta", fail_after_chunks=2),
        provider("b", responder=lambda m: "complete answer here"),
    ])
    evs = await collect(c.gateway, [HumanMessage("x")])
    types = [e.type for e in evs]
    assert "reset" in types
    after_reset = "".join(e.text for e in evs[types.index("reset") + 1:] if e.type == "token")
    assert after_reset == "complete answer here"


def test_build_providers_only_with_keys(tmp_path):
    s = make_settings(tmp_path, OPEN_ROUTER_API_KEY="k1", OPEN_ROUTER_FALLBACK_API_KEY="k2", GROQ_API_KEY="k3", MISTRAL_API_KEY="k4")
    names = [p.name for p in build_providers(s)]
    assert names == ["openrouter", "openrouter-backup", "groq", "mistral"]
    s2 = make_settings(tmp_path, GROQ_API_KEY="k3")
    assert [p.name for p in build_providers(s2)] == ["groq"]
