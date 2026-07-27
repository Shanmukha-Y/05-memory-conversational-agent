"""Session.close() must flush any verbatim (never-compressed) trailing
turns into long-term memory. Without this, whatever was said after the
last compression trigger is silently lost the moment the process exits --
exactly the failure mode that let a late-conversation correction
("we've switched to Qdrant") never make it into the store. All LLM calls
mocked -- no network.
"""

import shutil
import tempfile
from pathlib import Path

import pytest

from memagent import extractor
from memagent.audit import AuditLog
from memagent.buffer import Buffer, Turn
from memagent.memory_store import MemoryStore
from memagent.schemas import ExtractedFacts, Fact, FactType
from memagent.session import Session


@pytest.fixture
def tmp_paths():
    tmp = Path(tempfile.mkdtemp())
    yield tmp / "chroma", tmp / "db.sqlite3"
    shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture
def tmp_session(tmp_paths):
    chroma_dir, sqlite_path = tmp_paths
    store = MemoryStore(chroma_dir=chroma_dir, sqlite_path=sqlite_path, embed_fn=lambda text: [1.0, 0.0])
    audit = AuditLog(sqlite_path=sqlite_path)
    yield Session(user_id="alice", store=store, audit=audit, buffer=Buffer())


def test_flush_extracts_never_compressed_trailing_turns(monkeypatch, tmp_session):
    monkeypatch.setattr(
        extractor.llm,
        "generate_json",
        lambda *a, **k: ExtractedFacts(
            facts=[Fact(text="User switched to Qdrant", type=FactType.decision, importance=5)]
        ),
    )
    # Simulate turns that were added but never compressed (buffer never
    # crossed its trigger threshold again before the session ended).
    tmp_session.buffer.add(Turn(role="user", content="We switched to Qdrant."))
    tmp_session.buffer.add(Turn(role="assistant", content="Got it, noted."))

    assert tmp_session.store.list_facts("alice") == []
    tmp_session.flush()
    facts = tmp_session.store.list_facts("alice")
    assert len(facts) == 1
    assert facts[0].text == "User switched to Qdrant"


def test_flush_skips_llm_call_when_buffer_is_all_summary(monkeypatch, tmp_session):
    called = []
    monkeypatch.setattr(extractor.llm, "generate_json", lambda *a, **k: called.append(1))
    tmp_session.buffer.add(Turn(role="summary", content="[compressed] earlier chat", compressed=True))

    tmp_session.flush()
    assert called == []  # summary-only buffer has nothing to extract


def test_flush_skips_llm_call_on_empty_buffer(monkeypatch, tmp_session):
    called = []
    monkeypatch.setattr(extractor.llm, "generate_json", lambda *a, **k: called.append(1))
    tmp_session.flush()
    assert called == []


def test_close_calls_flush_before_closing_store(monkeypatch, tmp_session, tmp_paths):
    monkeypatch.setattr(
        extractor.llm,
        "generate_json",
        lambda *a, **k: ExtractedFacts(facts=[Fact(text="User is Alice", type=FactType.identity, importance=5)]),
    )
    tmp_session.buffer.add(Turn(role="user", content="I'm Alice."))
    tmp_session.buffer.add(Turn(role="assistant", content="Hi Alice!"))

    tmp_session.close()
    # Re-open a fresh store against the same on-disk files to confirm the
    # flushed fact actually persisted before the connection was closed.
    chroma_dir, sqlite_path = tmp_paths
    reopened = MemoryStore(chroma_dir=chroma_dir, sqlite_path=sqlite_path, embed_fn=lambda text: [1.0, 0.0])
    facts = reopened.list_facts("alice")
    assert any(f.text == "User is Alice" for f in facts)
