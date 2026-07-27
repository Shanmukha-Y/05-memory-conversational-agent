"""Live, end-to-end proof of cross-session memory. Marked integration so it
never runs by default -- it hits a real (shared) Ollama server and can take
many minutes when the server is under load from other builders.

Session A plants facts across enough turns to force >=2 compressions.
"Restart" is simulated the same way session.py documents it should be: a
brand-new Session with a brand-new (empty) Buffer, pointed at the same
on-disk MemoryStore/AuditLog. If session B can still answer questions that
were only ever said in session A, the memory is genuinely in the store, not
carried by the process.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest

from memagent.audit import AuditLog
from memagent.buffer import Buffer
from memagent.memory_store import MemoryStore
from memagent.session import Session

# Small budget so this ~12-turn scripted conversation crosses the
# compression trigger more than once without needing a 50-turn script.
#
# 350 (an earlier value here) was too tight: real exchanges in this script
# run ~90-160 tokens each, so a 280-token trigger threshold fired on almost
# every turn -- 9 compressions across 12 turns, observed live. That's
# pathological: each compression re-summarizes the *previous* summary
# turn along with new content, and repeatedly summarizing an
# already-terse summary measurably drives the model to embellish
# plausible-sounding specifics that were never said (observed live: a
# stored fact inventing "Recall@K" and "MRR" metrics out of nothing).
# 600 (threshold 480) still reliably forces >=2 real compressions across
# this script without re-summarizing on nearly every turn.
BUFFER_BUDGET = 600

SESSION_A_TURNS = [
    "Hi, I'm Alice, a backend engineer at a logistics startup.",
    "I'm building a RAG system for our internal knowledge base.",
    "We picked ChromaDB as the vector store to start.",
    "My favorite programming language is Rust, though most of my day job is Python.",
    "We decided to ship the first version behind a feature flag instead of a full rollout.",
    "Random aside: what's a fun fact about narwhals?",
    "Haha nice. Anyway, my manager Priya has been great about giving us room to experiment.",
    "We're targeting a demo in three weeks for the leadership team.",
    "Update: we actually switched from ChromaDB to Qdrant, it handles our scale much better.",
    "One more preference: I like being addressed casually, not formally.",
    "Just to be fully clear: we are NOT using ChromaDB anymore, we're fully on Qdrant now.",
    "That's everything for today, thanks!",
]

SESSION_B_QUESTIONS = [
    ("What's my name and what's my job?", ["Alice", "backend engineer"]),
    ("What project am I working on?", ["RAG", "knowledge base"]),
    ("What vector store am I using right now?", ["Qdrant"]),
    ("What did we decide about the rollout?", ["feature flag"]),
    ("What's my manager's name?", ["Priya"]),
]


@pytest.mark.integration
@pytest.mark.timeout(1800)
def test_two_session_cross_restart_recall():
    tmp = Path(tempfile.mkdtemp())
    try:
        chroma_dir = tmp / "chroma"
        sqlite_path = tmp / "db.sqlite3"
        user_id = "integration_alice"

        # --- Session A: plant facts, force compression ---
        store_a = MemoryStore(chroma_dir=chroma_dir, sqlite_path=sqlite_path)
        audit_a = AuditLog(sqlite_path=sqlite_path)
        session_a = Session(
            user_id=user_id,
            store=store_a,
            audit=audit_a,
            buffer=Buffer(budget_tokens=BUFFER_BUDGET, trigger_ratio=0.8),
        )

        for turn in SESSION_A_TURNS:
            session_a.handle_turn(turn)

        compression_events = audit_a.events(user_id, "compression")
        session_a.close()

        assert len(compression_events) >= 2, (
            f"expected >=2 compressions to have fired during session A, got {len(compression_events)}"
        )

        facts_after_a = MemoryStore(chroma_dir=chroma_dir, sqlite_path=sqlite_path).list_facts(user_id)
        assert len(facts_after_a) >= 3, "expected several distinct facts extracted during session A"

        # --- "Restart": brand-new Session, brand-new empty Buffer, same store ---
        store_b = MemoryStore(chroma_dir=chroma_dir, sqlite_path=sqlite_path)
        audit_b = AuditLog(sqlite_path=sqlite_path)
        session_b = Session(user_id=user_id, store=store_b, audit=audit_b, buffer=Buffer())

        assert session_b.buffer.turns == [], "restart must not restore anything into the short-term buffer"

        hits = 0
        answers = []
        for question, expected_keywords in SESSION_B_QUESTIONS:
            answer = session_b.handle_turn(question)
            answers.append((question, answer))
            if any(k.lower() in answer.lower() for k in expected_keywords):
                hits += 1

        session_b.close()

        assert hits >= 4, (
            f"expected >=4/5 correct recalls from long-term memory alone, got {hits}/5\n"
            + "\n".join(f"Q: {q}\nA: {a}" for q, a in answers)
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
