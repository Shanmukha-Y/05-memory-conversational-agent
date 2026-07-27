"""Compress the oldest half of the buffer into one summary turn once the
buffer crosses its trigger threshold. Nothing is truly lost: the turns
being compressed are already in the SQLite transcript log (written by
session.py before compression ever runs) -- this only shrinks what rides
along in every subsequent prompt.
"""

from __future__ import annotations

from dataclasses import dataclass

from memagent import llm
from memagent.buffer import Buffer, Turn
from memagent.config import CONFIG

SUMMARY_SYSTEM_PROMPT = """You compress older turns of a conversation into \
a short third-person recap that preserves anything a later reply might \
need: names, decisions, ongoing projects, preferences, and open threads. \
Omit small talk and pleasantries. Only include what was actually said --\
do not invent tools, metrics, terminology, or plausible-sounding details \
that never came up, even to make the summary read more smoothly. Write \
2-5 sentences, no preamble."""


@dataclass
class CompressionResult:
    summary: str
    compressed_turns: list[Turn]


def _render_turns(turns: list[Turn]) -> str:
    lines = []
    for t in turns:
        speaker = "User" if t.role == "user" else "Assistant"
        lines.append(f"{speaker}: {t.content}")
    return "\n".join(lines)


def compress(buffer: Buffer, keep_last_n: int = CONFIG.keep_last_n_turns) -> CompressionResult | None:
    """Summarize the oldest half of `buffer.turns`, keeping at least
    `keep_last_n` most recent turns verbatim, and mutate `buffer` in place
    to hold [summary_turn, *kept_turns]. Returns None (no-op) if the buffer
    is too small to usefully compress."""
    turns = buffer.turns
    if len(turns) <= keep_last_n:
        return None

    split = max(1, len(turns) // 2)
    split = min(split, len(turns) - keep_last_n)
    if split <= 0:
        return None

    to_compress = turns[:split]
    to_keep = turns[split:]

    transcript = _render_turns(to_compress)
    prompt = f"Conversation to compress:\n{transcript}"
    summary_text = llm.chat(
        [
            {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=CONFIG.compression_temperature,
    )

    summary_turn = Turn(role="summary", content=f"[compressed] {summary_text.strip()}", compressed=True)
    buffer.turns = [summary_turn, *to_keep]

    return CompressionResult(summary=summary_text.strip(), compressed_turns=to_compress)
