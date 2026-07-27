"""Pull durable facts about the user/their work out of turns being
compressed. Chit-chat that carries no lasting information about the user
should yield an empty fact list -- the extraction prompt is deliberately
conservative about this."""

from __future__ import annotations

from memagent import llm
from memagent.buffer import Turn
from memagent.config import CONFIG
from memagent.schemas import ExtractedFacts, Fact

EXTRACTION_SYSTEM_PROMPT = """You extract durable facts about the USER \
(not the assistant) from a conversation snippet: who they are, what \
they're working on, their preferences, and decisions they've made. \
Ignore greetings, small talk, and anything that won't matter in a future \
conversation. Each fact must be a short, self-contained, third-person \
statement (e.g. "User is building a RAG system", not "I'm building..."). \
Rate importance 1 (trivial) to 5 (core identity/critical decision). If \
there is nothing durable, return an empty list.

Respond with JSON: {"facts": [{"text": string, "type": \
"identity"|"preference"|"project"|"decision", "importance": 1-5}, ...]}."""


def _render_turns(turns: list[Turn]) -> str:
    lines = []
    for t in turns:
        speaker = "User" if t.role == "user" else "Assistant"
        lines.append(f"{speaker}: {t.content}")
    return "\n".join(lines)


def extract_facts(turns: list[Turn]) -> list[Fact]:
    """Extract facts from a batch of turns (typically the ones just
    compressed out of the buffer)."""
    user_turns = [t for t in turns if t.role in ("user", "assistant")]
    if not user_turns:
        return []

    transcript = _render_turns(user_turns)
    prompt = f"Conversation:\n{transcript}"
    result: ExtractedFacts = llm.generate_json(
        prompt,
        schema=ExtractedFacts,
        system=EXTRACTION_SYSTEM_PROMPT,
        temperature=CONFIG.extraction_temperature,
    )
    return result.facts
