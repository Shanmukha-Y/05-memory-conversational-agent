"""Short-term rolling buffer: the live conversation, token-budgeted.

Token accounting is a cheap heuristic (chars / 4) rather than a real
tokenizer -- good enough to decide "is it time to compress" without an
extra dependency, and the trigger fires well before the hard budget so the
approximation's error margin never matters in practice.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from memagent.config import CONFIG

Role = str  # "user" | "assistant" | "summary"


def estimate_tokens(text: str) -> int:
    """Rough token count: ~4 chars/token, minimum 1 for any non-empty text."""
    if not text:
        return 0
    return max(1, len(text) // 4)


@dataclass
class Turn:
    role: Role
    content: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    compressed: bool = False

    def tokens(self) -> int:
        return estimate_tokens(self.content)


class Buffer:
    """In-memory rolling turn list. Never persisted -- a process restart
    starts with an empty buffer; long-term recall is what carries memory
    across restarts, not this object."""

    def __init__(
        self,
        budget_tokens: int = CONFIG.buffer_budget_tokens,
        trigger_ratio: float = CONFIG.compression_trigger_ratio,
    ) -> None:
        self.turns: list[Turn] = []
        self.budget_tokens = budget_tokens
        self.trigger_ratio = trigger_ratio

    def add(self, turn: Turn) -> None:
        self.turns.append(turn)

    def add_message(self, role: Role, content: str) -> Turn:
        turn = Turn(role=role, content=content)
        self.add(turn)
        return turn

    def total_tokens(self) -> int:
        return sum(t.tokens() for t in self.turns)

    def trigger_threshold(self) -> int:
        return int(self.budget_tokens * self.trigger_ratio)

    def is_over_budget(self) -> bool:
        return self.total_tokens() >= self.trigger_threshold()

    def render(self) -> list[dict[str, str]]:
        """Buffer contents as chat messages. Summary turns are surfaced as
        an assistant-authored recap so the model treats them as established
        context rather than a system instruction."""
        rendered = []
        for t in self.turns:
            role = "assistant" if t.role == "summary" else t.role
            rendered.append({"role": role, "content": t.content})
        return rendered
