"""Dedup/contradiction decision paths in memory_store.upsert_fact, with the
embedding model and dedup-decision LLM call both mocked out. No network.

Embeddings are faked as simple 2D unit vectors so cosine similarity is
fully controlled: vectors pointing the same direction => similarity 1.0
(above threshold, triggers the dedup decider); orthogonal vectors =>
similarity 0.0 (below threshold, inserts as new).
"""

import shutil
import tempfile
from pathlib import Path

import pytest

from memagent.memory_store import MemoryStore
from memagent.schemas import DedupAction, DedupDecision, Fact, FactType


@pytest.fixture
def tmp_store():
    tmp = Path(tempfile.mkdtemp())

    def fake_embed(text: str) -> list[float]:
        # Same direction for anything containing "location", orthogonal
        # for anything containing "preference" -- lets tests control
        # similarity deterministically without a real model.
        if "PREF" in text:
            return [0.0, 1.0]
        return [1.0, 0.0]

    store = MemoryStore(
        chroma_dir=tmp / "chroma",
        sqlite_path=tmp / "db.sqlite3",
        dedup_threshold=0.65,
        embed_fn=fake_embed,
    )
    yield store
    store.close()
    shutil.rmtree(tmp, ignore_errors=True)


def test_first_fact_always_inserts(tmp_store):
    result = tmp_store.upsert_fact("alice", Fact(text="lives in NYC", type=FactType.identity, importance=3))
    assert result.action == "insert"
    assert result.record.text == "lives in NYC"


def test_dissimilar_fact_inserts_without_calling_decider(tmp_store):
    calls = []
    tmp_store.dedup_decider = lambda existing, new: calls.append((existing, new)) or DedupDecision(action=DedupAction.skip)

    tmp_store.upsert_fact("alice", Fact(text="lives in NYC", type=FactType.identity, importance=3))
    result = tmp_store.upsert_fact("alice", Fact(text="PREF dark mode", type=FactType.preference, importance=2))

    assert result.action == "insert"
    assert calls == []  # decider never invoked for orthogonal embedding
    assert len(tmp_store.list_facts("alice")) == 2


def test_similar_fact_replace_updates_in_place(tmp_store):
    tmp_store.dedup_decider = lambda existing, new: DedupDecision(action=DedupAction.replace, reasoning="contradicts")

    first = tmp_store.upsert_fact("alice", Fact(text="lives in NYC", type=FactType.identity, importance=3))
    second = tmp_store.upsert_fact("alice", Fact(text="moved to Boston", type=FactType.identity, importance=4))

    assert second.action == "replace"
    assert second.previous_text == "lives in NYC"
    assert second.record.id == first.record.id  # same row, not a new one
    assert second.record.text == "moved to Boston"
    assert second.record.importance == 4  # max(3, 4)

    facts = tmp_store.list_facts("alice")
    assert len(facts) == 1  # not duplicated
    assert facts[0].text == "moved to Boston"


def test_similar_fact_merge_combines_text(tmp_store):
    tmp_store.dedup_decider = lambda existing, new: DedupDecision(
        action=DedupAction.merge, merged_text="lives in NYC, working remotely", reasoning="complementary"
    )

    first = tmp_store.upsert_fact("alice", Fact(text="lives in NYC", type=FactType.identity, importance=3))
    second = tmp_store.upsert_fact("alice", Fact(text="works remotely", type=FactType.identity, importance=2))

    assert second.action == "merge"
    assert second.record.id == first.record.id
    assert second.record.text == "lives in NYC, working remotely"

    facts = tmp_store.list_facts("alice")
    assert len(facts) == 1


def test_similar_fact_skip_leaves_existing_untouched(tmp_store):
    tmp_store.dedup_decider = lambda existing, new: DedupDecision(action=DedupAction.skip, reasoning="redundant")

    first = tmp_store.upsert_fact("alice", Fact(text="lives in NYC", type=FactType.identity, importance=3))
    second = tmp_store.upsert_fact("alice", Fact(text="based in New York City", type=FactType.identity, importance=3))

    assert second.action == "skip"
    facts = tmp_store.list_facts("alice")
    assert len(facts) == 1
    assert facts[0].text == "lives in NYC"
    assert facts[0].updated_at == first.record.updated_at


def test_forgotten_fact_excluded_from_dedup_matching(tmp_store):
    """A forgotten (inactive) fact shouldn't be resurrected by dedup logic
    matching against it."""
    tmp_store.dedup_decider = lambda existing, new: DedupDecision(action=DedupAction.replace)
    first = tmp_store.upsert_fact("alice", Fact(text="lives in NYC", type=FactType.identity, importance=3))
    tmp_store.forget("alice", first.record.id)

    result = tmp_store.upsert_fact("alice", Fact(text="lives in Boston", type=FactType.identity, importance=3))
    assert result.action == "insert"
    active_facts = tmp_store.list_facts("alice")
    assert len(active_facts) == 1
    assert active_facts[0].text == "lives in Boston"
