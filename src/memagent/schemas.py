"""Pydantic schemas for LLM-structured output (extraction, dedup decisions)."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class FactType(str, Enum):
    identity = "identity"
    preference = "preference"
    project = "project"
    decision = "decision"


class Fact(BaseModel):
    """A single durable fact about the user or their work, pulled from a
    turn of conversation being compressed out of the short-term buffer."""

    text: str = Field(min_length=1, description="Self-contained statement of the fact.")
    type: FactType
    importance: int = Field(ge=1, le=5)


class ExtractedFacts(BaseModel):
    """Output of the fact extractor. Empty list is valid (pure chit-chat)."""

    facts: list[Fact] = Field(default_factory=list)


class DedupAction(str, Enum):
    replace = "replace"
    merge = "merge"
    skip = "skip"


class DedupDecision(BaseModel):
    """Decision for a new fact that is highly similar to an existing one."""

    action: DedupAction
    merged_text: str | None = Field(
        default=None,
        description="Required when action=merge: the combined fact text.",
    )
    reasoning: str = ""
