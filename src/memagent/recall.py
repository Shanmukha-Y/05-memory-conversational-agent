"""Long-term recall: query the vector store, rescore by recency and
importance, return the top-k memories to inject into context.

Pure similarity ranking fails for this use case: a fact that was true
months ago but happens to be semantically close to the current question
will outrank a fresher, more relevant one. The score blends three signals
multiplicatively so that any one of them going to zero kills the memory's
relevance:

    score = similarity * exp(-age_days / half_life_days) * (importance / 5)

`score_memory` is a pure function (no I/O) so the formula's ordering
guarantees can be unit-tested without a running LLM or vector store.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone

from memagent.config import CONFIG
from memagent.memory_store import MemoryRecord, MemoryStore


def score_memory(
    similarity: float,
    age_days: float,
    importance: int,
    half_life_days: float = CONFIG.recall_half_life_days,
) -> float:
    """similarity in [0,1]-ish (cosine can dip slightly negative), age_days
    >= 0, importance in [1,5]. Returns a non-negative relevance score."""
    if half_life_days <= 0:
        decay = 1.0
    else:
        decay = math.exp(-age_days / half_life_days)
    return max(0.0, similarity) * decay * (importance / 5)


@dataclass
class RecalledMemory:
    record: MemoryRecord
    similarity: float
    age_days: float
    decay: float
    score: float


def recall(
    store: MemoryStore,
    user_id: str,
    query_text: str,
    top_k: int = CONFIG.recall_top_k,
    half_life_days: float = CONFIG.recall_half_life_days,
    candidate_k: int = CONFIG.recall_candidate_k,
) -> list[RecalledMemory]:
    """Embed `query_text`, pull nearest-neighbor candidates for `user_id`
    from the vector store, rescore by recency+importance, and return the
    top_k, highest score first."""
    candidates = store.query_candidates(user_id, query_text, n=candidate_k)
    now = datetime.now(timezone.utc)

    recalled: list[RecalledMemory] = []
    for record, similarity in candidates:
        if not record.active:
            continue
        age_days = max(0.0, (now - record.updated_at).total_seconds() / 86400)
        decay = math.exp(-age_days / half_life_days) if half_life_days > 0 else 1.0
        score = score_memory(similarity, age_days, record.importance, half_life_days)
        recalled.append(
            RecalledMemory(record=record, similarity=similarity, age_days=age_days, decay=decay, score=score)
        )

    recalled.sort(key=lambda r: r.score, reverse=True)
    return recalled[:top_k]


def format_recall_block(recalled: list[RecalledMemory]) -> str:
    """Render recalled memories as the 'Known about this user:' context
    block injected into the system prompt."""
    if not recalled:
        return ""
    lines = ["Known about this user (from long-term memory):"]
    for r in recalled:
        lines.append(f"- {r.record.text}")
    return "\n".join(lines)
