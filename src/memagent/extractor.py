"""Pull durable facts about the user/their work out of turns being
compressed. Chit-chat that carries no lasting information about the user
should yield an empty fact list -- the extraction prompt is deliberately
conservative about this.

Extraction source is restricted to the USER's own turns -- assistant turns
are deliberately excluded, not just summary turns. This was found live,
not assumed: a chat reply like "operational docs and SOPs" or "Recall@K
and MRR" is the assistant riffing with plausible-sounding domain color that
the user never said, and treating assistant turns as equally-trustworthy
extraction source material let that riffing get stored as a "fact about
the user." Summary turns were already excluded (role == "summary"), but
that filter never touched this bug -- the contamination came from live
assistant chat replies, not from summarized text. See readme.html for the
full traced example."""

from __future__ import annotations

from memagent import llm
from memagent.buffer import Turn
from memagent.config import CONFIG
from memagent.schemas import ExtractedFacts, Fact

EXTRACTION_SYSTEM_PROMPT = """You extract durable facts about the USER \
from a set of things the USER said: who they are, what they're working \
on, their preferences, and decisions they've made. Ignore greetings, \
small talk, and anything that won't matter in a future conversation. \
Each fact must be a short, self-contained, third-person statement (e.g. \
"User is building a RAG system", not "I'm building..."). Only state what \
was actually said -- do not add plausible-sounding details, tools, \
metrics, or terminology that weren't explicitly mentioned. Rate \
importance 1 (trivial) to 5 (core identity/critical decision). If there \
is nothing durable, return an empty list.

Respond with JSON: {"facts": [{"text": string, "type": \
"identity"|"preference"|"project"|"decision", "importance": 1-5}, ...]}."""


def _render_turns(turns: list[Turn]) -> str:
    return "\n".join(f"User: {t.content}" for t in turns)


def extract_facts(turns: list[Turn]) -> list[Fact]:
    """Extract facts from a batch of turns (typically the ones just
    compressed out of the buffer). Only the user's own turns are used as
    source material -- see module docstring for why."""
    source_turns = [t for t in turns if t.role == "user"]
    if not source_turns:
        return []

    transcript = _render_turns(source_turns)
    prompt = f"Things the user said:\n{transcript}"
    result: ExtractedFacts = llm.generate_json(
        prompt,
        schema=ExtractedFacts,
        system=EXTRACTION_SYSTEM_PROMPT,
        temperature=CONFIG.extraction_temperature,
    )
    return result.facts
