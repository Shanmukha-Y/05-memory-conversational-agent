"""Long-term memory store: ChromaDB (vectors) + SQLite (authoritative
metadata) keyed by user_id, with dedup/contradiction handling on upsert.

Chroma holds the embedding + text per fact, keyed by the same UUID as the
SQLite row so the two never drift. SQLite is authoritative for importance,
timestamps, and the active flag -- Chroma is purely an ANN index over text.

Dedup: when a newly extracted fact embeds within `dedup_similarity_threshold`
cosine similarity of an existing fact for that user, the two are handed to
an LLM (JSON mode) which decides one of:
  - skip:    new fact adds nothing -> existing row untouched
  - replace: new fact supersedes the old one (e.g. "moved to Boston"
             superseding "lives in NYC") -> row's text is overwritten,
             not duplicated
  - merge:   the two combine into one richer fact -> row's text becomes
             the LLM's merged text
Either way `updated_at` is bumped, which is what gives a reaffirmed/updated
fact a recency boost on the next recall.
"""

from __future__ import annotations

import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import chromadb

from memagent import llm
from memagent.config import CONFIG
from memagent.schemas import DedupAction, DedupDecision, Fact


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_dt(value: str) -> datetime:
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


@dataclass
class MemoryRecord:
    id: str
    user_id: str
    text: str
    type: str
    importance: int
    created_at: datetime
    updated_at: datetime
    active: bool = True


@dataclass
class UpsertResult:
    action: str  # "insert" | "replace" | "merge" | "skip"
    record: MemoryRecord | None
    previous_text: str | None = None


DEDUP_SYSTEM_PROMPT = """You reconcile a newly observed fact about a user \
against an existing, highly-similar fact already stored about them. Decide \
exactly one action:
- "replace": the new fact supersedes the old one (e.g. the user moved, \
switched tools/decisions, or otherwise contradicts the old fact). The old \
fact is no longer true.
- "merge": the two facts are complementary and should be combined into one \
richer statement (set merged_text to the combined fact).
- "skip": the new fact says nothing the existing fact doesn't already cover.

Respond with JSON: {"action": "replace"|"merge"|"skip", "merged_text": \
string or null, "reasoning": string}."""


def decide_dedup(existing_text: str, new_text: str) -> DedupDecision:
    """Default dedup decider: one live LLM call in JSON mode. Tests should
    monkeypatch this function (or pass a custom `dedup_decider` to
    MemoryStore) rather than hitting the network."""
    prompt = f'Existing fact: "{existing_text}"\nNew fact: "{new_text}"'
    result = llm.generate_json(
        prompt,
        schema=DedupDecision,
        system=DEDUP_SYSTEM_PROMPT,
        temperature=CONFIG.dedup_temperature,
    )
    return result


class MemoryStore:
    def __init__(
        self,
        chroma_dir: Path = CONFIG.chroma_dir,
        sqlite_path: Path = CONFIG.sqlite_path,
        dedup_threshold: float = CONFIG.dedup_similarity_threshold,
        dedup_decider=decide_dedup,
        embed_fn=None,
    ) -> None:
        Path(chroma_dir).parent.mkdir(parents=True, exist_ok=True)
        Path(sqlite_path).parent.mkdir(parents=True, exist_ok=True)
        self._chroma = chromadb.PersistentClient(path=str(chroma_dir))
        self._conn = sqlite3.connect(str(sqlite_path))
        self._conn.row_factory = sqlite3.Row
        self.dedup_threshold = dedup_threshold
        self.dedup_decider = dedup_decider
        self.embed_fn = embed_fn or llm.embed
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS facts (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                text TEXT NOT NULL,
                type TEXT NOT NULL,
                importance INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                active INTEGER NOT NULL DEFAULT 1
            )
            """
        )
        self._conn.commit()

    def _collection(self, user_id: str):
        return self._chroma.get_or_create_collection(
            name=f"memagent_{user_id}",
            metadata={"hnsw:space": "cosine"},
        )

    def _row_to_record(self, row: sqlite3.Row) -> MemoryRecord:
        return MemoryRecord(
            id=row["id"],
            user_id=row["user_id"],
            text=row["text"],
            type=row["type"],
            importance=row["importance"],
            created_at=_parse_dt(row["created_at"]),
            updated_at=_parse_dt(row["updated_at"]),
            active=bool(row["active"]),
        )

    def get_fact(self, user_id: str, fact_id: str) -> MemoryRecord | None:
        row = self._conn.execute(
            "SELECT * FROM facts WHERE id = ? AND user_id = ?", (fact_id, user_id)
        ).fetchone()
        return self._row_to_record(row) if row else None

    def list_facts(self, user_id: str, active_only: bool = True) -> list[MemoryRecord]:
        query = "SELECT * FROM facts WHERE user_id = ?"
        params: tuple = (user_id,)
        if active_only:
            query += " AND active = 1"
        query += " ORDER BY updated_at DESC"
        rows = self._conn.execute(query, params).fetchall()
        return [self._row_to_record(r) for r in rows]

    def forget(self, user_id: str, fact_id: str) -> bool:
        """Deactivate a fact and remove it from the vector index so it can
        never be recalled again. Row is kept (active=0) for audit history."""
        record = self.get_fact(user_id, fact_id)
        if record is None:
            return False
        self._conn.execute(
            "UPDATE facts SET active = 0, updated_at = ? WHERE id = ?",
            (_now().isoformat(), fact_id),
        )
        self._conn.commit()
        try:
            self._collection(user_id).delete(ids=[fact_id])
        except Exception:
            pass
        return True

    def _insert(self, user_id: str, fact: Fact, embedding: list[float]) -> MemoryRecord:
        fact_id = str(uuid.uuid4())
        now = _now()
        self._conn.execute(
            "INSERT INTO facts (id, user_id, text, type, importance, created_at, updated_at, active) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, 1)",
            (fact_id, user_id, fact.text, fact.type.value, fact.importance, now.isoformat(), now.isoformat()),
        )
        self._conn.commit()
        self._collection(user_id).add(
            ids=[fact_id],
            embeddings=[embedding],
            documents=[fact.text],
            metadatas=[{"type": fact.type.value}],
        )
        return MemoryRecord(
            id=fact_id,
            user_id=user_id,
            text=fact.text,
            type=fact.type.value,
            importance=fact.importance,
            created_at=now,
            updated_at=now,
            active=True,
        )

    def _update_text(self, user_id: str, fact_id: str, new_text: str, importance: int, embedding: list[float]) -> MemoryRecord:
        now = _now()
        self._conn.execute(
            "UPDATE facts SET text = ?, importance = ?, updated_at = ? WHERE id = ?",
            (new_text, importance, now.isoformat(), fact_id),
        )
        self._conn.commit()
        self._collection(user_id).update(ids=[fact_id], embeddings=[embedding], documents=[new_text])
        return self.get_fact(user_id, fact_id)

    def query_candidates(self, user_id: str, query_text: str, n: int) -> list[tuple[MemoryRecord, float]]:
        """Nearest-neighbor facts for `query_text`, as (record, similarity)
        pairs. similarity = 1 - cosine_distance."""
        coll = self._collection(user_id)
        count = coll.count()
        if count == 0:
            return []
        embedding = self.embed_fn(query_text)
        results = coll.query(query_embeddings=[embedding], n_results=min(n, count))
        ids = results["ids"][0]
        distances = results["distances"][0]
        pairs = []
        for fact_id, distance in zip(ids, distances):
            record = self.get_fact(user_id, fact_id)
            if record is None:
                continue
            similarity = 1.0 - distance
            pairs.append((record, similarity))
        return pairs

    def upsert_fact(self, user_id: str, fact: Fact) -> UpsertResult:
        """Insert a new fact, or reconcile it against a highly-similar
        existing one via `dedup_decider`."""
        embedding = self.embed_fn(fact.text)
        coll = self._collection(user_id)

        if coll.count() > 0:
            results = coll.query(query_embeddings=[embedding], n_results=1)
            best_id = results["ids"][0][0] if results["ids"][0] else None
            best_distance = results["distances"][0][0] if results["distances"][0] else None
            if best_id is not None and best_distance is not None:
                similarity = 1.0 - best_distance
                if similarity >= self.dedup_threshold:
                    existing = self.get_fact(user_id, best_id)
                    if existing is not None and existing.active:
                        decision: DedupDecision = self.dedup_decider(existing.text, fact.text)
                        if decision.action == DedupAction.skip:
                            return UpsertResult(action="skip", record=existing, previous_text=existing.text)
                        if decision.action == DedupAction.replace:
                            previous_text = existing.text
                            new_embedding = self.embed_fn(fact.text)
                            record = self._update_text(
                                user_id, existing.id, fact.text, max(existing.importance, fact.importance), new_embedding
                            )
                            return UpsertResult(action="replace", record=record, previous_text=previous_text)
                        if decision.action == DedupAction.merge:
                            merged_text = decision.merged_text or fact.text
                            previous_text = existing.text
                            new_embedding = self.embed_fn(merged_text)
                            record = self._update_text(
                                user_id, existing.id, merged_text, max(existing.importance, fact.importance), new_embedding
                            )
                            return UpsertResult(action="merge", record=record, previous_text=previous_text)

        record = self._insert(user_id, fact, embedding)
        return UpsertResult(action="insert", record=record)

    def close(self) -> None:
        self._conn.close()
