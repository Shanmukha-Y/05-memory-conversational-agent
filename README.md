# memagent

Project 05 of a portfolio of small, focused agent builds: a conversational agent with long-term memory that survives process restarts, demonstrating two-tier memory management (rolling buffer + compressed/distilled recall) for production agents.

## What it does

- **Two-tier memory**: a token-budgeted rolling buffer (~2,000 tokens) holds the live conversation verbatim; a persistent ChromaDB + SQLite store holds distilled facts that survive the process dying.
- **Compression, not deletion**: once the buffer hits 80% of budget, the oldest half is summarized into one `[compressed]` turn — the raw turns still live in the SQLite transcript log and are never lost.
- **Fact extraction**: compressed-out content is passed to `qwen3.5:9b` in JSON mode to pull durable facts (identity, preference, project, decision), validated against a Pydantic schema.
- **Scored recall**: `similarity × recency_decay × importance` ranks candidate facts, so a fresher or more important fact can beat a merely-similar stale one — covered by exact-ordering unit tests.
- **Contradiction handling**: new facts are embedded and compared against existing ones; above a measured similarity threshold, an LLM call decides replace/merge/skip so memory updates in place instead of accumulating duplicates.
- **`/why` transparency**: every recall shows its full score breakdown (similarity, age, decay, importance, final score) — nothing about "what the agent remembers" is a black box.

## Quick start

```bash
uv sync

# requires Ollama running locally with both models pulled:
#   ollama pull qwen3.5:9b
#   ollama pull nomic-embed-text

uv run mem chat --user alice
```

Fast tests (unit + mocked-LLM, no network, run in well under a second):

```bash
uv run pytest
```

Live integration test — hits a real local Ollama server running `qwen3.5:9b` and `nomic-embed-text`, excluded by default:

```bash
uv run pytest -m integration -v
```

Recall eval:

```bash
uv run python eval/run_eval.py --limit 2 --verbose
```

## Learnings

**The dedup threshold was wrong by intuition, right by measurement.** A "plausible" cosine-similarity cutoff of 0.9 for detecting duplicate facts was tested against real `nomic-embed-text` embeddings and turned out to be actively dangerous: genuine contradictions about the same fact ("lives in NYC" → "moved to Boston": 0.714 cosine; "uses Chroma" → "switched to Qdrant": 0.791) scored *lower* than a 0.9 bar, while true paraphrases scored 0.906. A 0.9 threshold would have let every contradiction sail through as a silent duplicate insert instead of triggering an update. The threshold was recalibrated to 0.65 — above the unrelated-pairs ceiling (~0.44), below the lowest observed contradiction score (~0.71).

**A fabricated-memory chain, traced end to end.** Live verification surfaced a stored fact — "User plans to use controlled rollout to gather ground truth for evaluation metrics like Recall@K or MRR" — that nobody ever said. Tracing it through the audit log and raw transcript found the exact mechanism: the fact extractor's turn filter originally included `role == "assistant"` alongside `role == "user"`. A chat model replying to "I'm building a RAG system for our internal knowledge base" naturally elaborates ("...especially for getting operational docs and SOPs into the pipeline") — normal chat behavior, but the extractor treated that elaboration as a fact about the user. The chain compounded across re-compression cycles: each summarization pass treated the prior cycle's invented detail as established truth and elaborated further (inventing a specific embedding model name, then specific eval metrics), until extraction independently reached the same fabricated territory and stored it permanently. Fix: extraction now sources exclusively from `role == "user"` turns; both extraction and summarization prompts were also tightened to explicitly forbid inventing plausible-sounding details. A dedicated test plants a fabricated detail in an assistant turn and asserts it never reaches the extraction prompt at all — not just that it's absent from the output. Residual, accepted limitation: short-term `[compressed]` summaries (session-local only, never persisted as long-term memory) can still drift over many re-compression cycles, since the summarizer necessarily sees both roles to track conversational continuity.

**Silent data loss from buffer-only extraction.** A live cross-session recall test failed because extraction only ran on compress-triggered batches — anything said after the last compression, right up to session end, sat verbatim in the buffer and vanished when the process exited. Root-caused via audit log + transcript correlation to a late-session correction ("we're fully on Qdrant now") that never got extracted. Fixed with `Session.flush()`, which extracts from any remaining verbatim turns on close.

**Buffer budget size affects more than speed.** The original 350-token integration/eval buffer budget was tight enough relative to real exchange sizes to trigger compression 9 times across a 12-turn script — which, beyond being slow, meant far more re-summarization cycles than necessary, directly increasing the surface area for the drift problem above. Retuned to 600 tokens.

Representative eval run: 9/10 facts recalled across two scripted dialogs (90%), 2/2 contradiction-handling cases passed, buffer stayed within budget in all runs. Numbers vary slightly turn-to-turn since generation isn't deterministic.

See `readme.html` for the full write-up, architecture diagram, and demo transcript.
