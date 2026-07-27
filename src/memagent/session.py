"""Session lifecycle for one user's conversation.

A restart creates a brand-new Session with an empty Buffer -- nothing about
past turns is restored into short-term memory. Whatever the agent "still
knows" after a restart comes entirely from `recall.recall()` querying the
persistent MemoryStore, which is the point: this class deliberately does
NOT reload prior buffer state, so it proves long-term memory lives in the
store, not the process.
"""

from __future__ import annotations

from memagent import compressor, extractor, llm, recall
from memagent.audit import AuditLog
from memagent.buffer import Buffer
from memagent.config import CONFIG
from memagent.memory_store import MemoryStore

SYSTEM_PROMPT = (
    "You are a helpful, concise conversational assistant with long-term "
    "memory of this user across sessions. When a 'Known about this user' "
    "block is present below, treat it as established fact and use it "
    "naturally -- don't ask the user to repeat things you already know."
)


class Session:
    def __init__(
        self,
        user_id: str,
        store: MemoryStore | None = None,
        audit: AuditLog | None = None,
        buffer: Buffer | None = None,
    ) -> None:
        self.user_id = user_id
        self.store = store or MemoryStore()
        self.audit = audit or AuditLog()
        self.buffer = buffer or Buffer()
        self.last_recall: list[recall.RecalledMemory] = []

    def handle_turn(self, user_message: str) -> str:
        self.audit.log_transcript(self.user_id, "user", user_message)

        recalled = recall.recall(
            self.store,
            self.user_id,
            user_message,
            top_k=CONFIG.recall_top_k,
            half_life_days=CONFIG.recall_half_life_days,
        )
        self.last_recall = recalled

        system_content = SYSTEM_PROMPT
        recall_block = recall.format_recall_block(recalled)
        if recall_block:
            system_content = f"{system_content}\n\n{recall_block}"

        messages = [{"role": "system", "content": system_content}]
        messages.extend(self.buffer.render())
        messages.append({"role": "user", "content": user_message})

        reply = llm.chat(messages, temperature=CONFIG.chat_temperature)

        self.buffer.add_message("user", user_message)
        self.buffer.add_message("assistant", reply)
        self.audit.log_transcript(self.user_id, "assistant", reply)

        if self.buffer.is_over_budget():
            self._compress_and_extract()

        return reply

    def _compress_and_extract(self) -> None:
        result = compressor.compress(self.buffer, keep_last_n=CONFIG.keep_last_n_turns)
        if result is None:
            return
        self.audit.log_compression(self.user_id, len(result.compressed_turns), result.summary)

        facts = extractor.extract_facts(result.compressed_turns)
        for fact in facts:
            upsert_result = self.store.upsert_fact(self.user_id, fact)
            self.audit.log_extraction(
                self.user_id, fact.text, upsert_result.action, upsert_result.previous_text
            )

    def flush(self) -> None:
        """Extract facts from any turns still sitting verbatim in the
        buffer -- i.e. turns added since the last compression, which never
        went through extraction because the buffer never crossed the
        trigger threshold again before the session ended.

        Without this, whatever was said in the last few turns of a session
        (which is exactly where a just-stated correction or decision is
        most likely to be) would be silently lost the moment the process
        exits: session B starts with an empty buffer by design, so nothing
        that never made it into the store is recoverable. `extract_facts`
        only draws from the user's own turns (see extractor.py), so this
        is safe to call even if the buffer is currently all-summary,
        assistant-only, or empty.
        """
        pending_user_turns = [t for t in self.buffer.turns if t.role == "user"]
        if not pending_user_turns:
            return
        facts = extractor.extract_facts(self.buffer.turns)
        for fact in facts:
            upsert_result = self.store.upsert_fact(self.user_id, fact)
            self.audit.log_extraction(
                self.user_id, fact.text, upsert_result.action, upsert_result.previous_text
            )

    def close(self) -> None:
        self.flush()
        self.store.close()
        self.audit.close()
