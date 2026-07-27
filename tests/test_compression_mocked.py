"""Compression + extraction pipeline with llm.chat / llm.generate_json
monkeypatched -- no network."""

from datetime import datetime, timezone

import pytest

from memagent import compressor, extractor, llm
from memagent.buffer import Buffer, Turn
from memagent.schemas import ExtractedFacts, Fact, FactType


def make_turns(n: int) -> list[Turn]:
    turns = []
    for i in range(n):
        role = "user" if i % 2 == 0 else "assistant"
        turns.append(Turn(role=role, content=f"turn {i} content", timestamp=datetime.now(timezone.utc)))
    return turns


def test_compress_below_keep_last_n_is_noop():
    buf = Buffer()
    buf.turns = make_turns(3)
    result = compressor.compress(buf, keep_last_n=4)
    assert result is None
    assert len(buf.turns) == 3


def test_compress_summarizes_oldest_half_and_keeps_rest(monkeypatch):
    calls = []

    def fake_chat(messages, temperature=0.0):
        calls.append(messages)
        return "User introduced themselves and described their project."

    monkeypatch.setattr(compressor.llm, "chat", fake_chat)

    buf = Buffer()
    buf.turns = make_turns(10)
    result = compressor.compress(buf, keep_last_n=4)

    assert result is not None
    assert len(result.compressed_turns) == 5  # oldest half of 10
    assert len(calls) == 1

    # buffer now: [summary_turn, *last 5 kept turns]
    assert len(buf.turns) == 6
    assert buf.turns[0].role == "summary"
    assert buf.turns[0].compressed is True
    assert buf.turns[0].content.startswith("[compressed]")
    assert buf.turns[1].content == "turn 5 content"  # first kept turn


def test_compress_never_eats_into_keep_last_n(monkeypatch):
    monkeypatch.setattr(compressor.llm, "chat", lambda messages, temperature=0.0: "summary")
    buf = Buffer()
    buf.turns = make_turns(5)
    result = compressor.compress(buf, keep_last_n=4)
    assert result is not None
    assert len(result.compressed_turns) == 1
    assert len(buf.turns) == 5  # 1 summary + 4 kept


def test_extract_facts_returns_pydantic_facts(monkeypatch):
    def fake_generate_json(prompt, schema, system=None, temperature=0.0, max_retries=1):
        return ExtractedFacts(
            facts=[
                Fact(text="User's name is Alice", type=FactType.identity, importance=5),
                Fact(text="User is building a RAG system", type=FactType.project, importance=4),
            ]
        )

    monkeypatch.setattr(extractor.llm, "generate_json", fake_generate_json)

    turns = [
        Turn(role="user", content="Hi, I'm Alice and I'm building a RAG system."),
        Turn(role="assistant", content="Nice to meet you Alice!"),
    ]
    facts = extractor.extract_facts(turns)
    assert len(facts) == 2
    assert facts[0].type == FactType.identity
    assert facts[1].importance == 4


def test_extract_facts_empty_turns_short_circuits_without_llm_call(monkeypatch):
    called = []
    monkeypatch.setattr(extractor.llm, "generate_json", lambda *a, **k: called.append(1))
    facts = extractor.extract_facts([])
    assert facts == []
    assert called == []


def test_extract_facts_can_return_empty_list_for_chitchat(monkeypatch):
    monkeypatch.setattr(
        extractor.llm, "generate_json", lambda *a, **k: ExtractedFacts(facts=[])
    )
    turns = [Turn(role="user", content="lol nice"), Turn(role="assistant", content="haha yeah")]
    facts = extractor.extract_facts(turns)
    assert facts == []


def test_extract_facts_never_sends_assistant_content_to_the_model(monkeypatch):
    """Regression test for a live-observed bug: the assistant's own
    conversational replies contain plausible-sounding embellishment the
    user never said (e.g. "operational docs and SOPs", "Recall@K and
    MRR"), and treating assistant turns as extraction source let that
    embellishment get stored as a "fact about the user." Assistant turns
    must never even reach the extraction prompt."""
    captured_prompts = []

    def fake_generate_json(prompt, schema, system=None, temperature=0.0, max_retries=1):
        captured_prompts.append(prompt)
        return ExtractedFacts(facts=[])

    monkeypatch.setattr(extractor.llm, "generate_json", fake_generate_json)

    turns = [
        Turn(role="user", content="I'm building a RAG system."),
        Turn(role="assistant", content="Nice, that probably involves Recall@K and MRR metrics, right?"),
    ]
    extractor.extract_facts(turns)

    assert len(captured_prompts) == 1
    assert "Recall@K" not in captured_prompts[0]
    assert "MRR" not in captured_prompts[0]
    assert "I'm building a RAG system" in captured_prompts[0]


def test_extract_facts_never_sends_summary_content_to_the_model(monkeypatch):
    """A planted fake specific that only exists in a prior [compressed]
    summary turn must never reach the extraction prompt either."""
    captured_prompts = []

    def fake_generate_json(prompt, schema, system=None, temperature=0.0, max_retries=1):
        captured_prompts.append(prompt)
        return ExtractedFacts(facts=[])

    monkeypatch.setattr(extractor.llm, "generate_json", fake_generate_json)

    turns = [
        Turn(role="summary", content="[compressed] User evaluates using text-embedding-3-large.", compressed=True),
        Turn(role="user", content="We picked ChromaDB as the vector store."),
    ]
    extractor.extract_facts(turns)

    assert len(captured_prompts) == 1
    assert "text-embedding-3-large" not in captured_prompts[0]
    assert "ChromaDB" in captured_prompts[0]


def test_extract_facts_source_is_user_turns_only(monkeypatch):
    monkeypatch.setattr(extractor.llm, "generate_json", lambda *a, **k: ExtractedFacts(facts=[]))
    turns = [
        Turn(role="assistant", content="only assistant content here"),
        Turn(role="summary", content="[compressed] only summary content here", compressed=True),
    ]
    # No user turns at all -> should short-circuit without an LLM call.
    called = []
    monkeypatch.setattr(extractor.llm, "generate_json", lambda *a, **k: called.append(1))
    facts = extractor.extract_facts(turns)
    assert facts == []
    assert called == []
