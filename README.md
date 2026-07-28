# memagent

Project 05 in a production-agent engineering series: a conversational agent with session memory and persistent fact recall, designed to make compression, provenance, contradiction handling, and memory failures observable.

## What it does

- **Two-tier memory.** A token-budgeted rolling buffer keeps the active conversation; ChromaDB and SQLite persist distilled facts and transcripts across process restarts.
- **Compression instead of silent deletion.** When the buffer crosses 80% of its budget, the oldest half is summarized into a `[compressed]` turn while the original turns remain in the SQLite transcript.
- **User-sourced fact extraction.** Only user-authored turns are eligible for persistent fact extraction. `qwen3.5:9b` emits typed identity, preference, project, and decision facts validated by Pydantic.
- **Scored recall.** Similarity, recency decay, and declared importance jointly rank candidates, with deterministic tests for ordering behavior.
- **Contradiction handling.** Similar existing facts trigger a model decision to replace, merge, or skip instead of accumulating silent duplicates.
- **Recall transparency.** `/why` exposes the similarity, age, decay, importance, and final score behind each recalled fact.
- **End-of-session flush.** Remaining user turns are extracted before close so facts said after the last compression cycle do not disappear on process exit.

## Quick start

```bash
uv sync
ollama pull qwen3.5:9b
ollama pull nomic-embed-text

uv run mem chat --user alice

# Fast unit and mocked-model suite
uv run pytest

# Live integration test
uv run pytest -m integration -v

# Small recall evaluation
uv run python eval/run_eval.py --limit 2 --verbose
```

## Privacy, tenancy, and poisoning boundary

This repository persists conversation transcripts and extracted facts in local SQLite and ChromaDB storage. It does not encrypt data at rest, authenticate users, isolate tenants, enforce retention periods, implement legal deletion workflows, redact secrets, or protect database files from another local process. `/why` is useful for debugging but can itself reveal sensitive memories. Use synthetic data unless those controls are added.

Restricting extraction to user-authored turns prevents assistant inventions from being promoted automatically into long-term facts; it does not make every user statement true or safe to retain. A user, imported transcript, or compromised upstream system can deliberately poison memory. The contradiction resolver and summarizer are model calls and can still make mistakes. Production memory should retain source identity and timestamp, distinguish assertions from verified facts, support user review and deletion, and apply policy before both storage and recall.

## Learnings

- **The duplicate threshold was unsafe when chosen by intuition.** A 0.90 cosine cutoff missed real contradictions that scored around 0.71–0.79, while true paraphrases scored around 0.91. Measured unrelated pairs remained below roughly 0.44, so the trigger was recalibrated to 0.65 for this corpus and embedding model.
- **A fabricated-memory chain was traced end to end.** The extractor originally consumed assistant as well as user turns. A plausible elaboration from the assistant was compressed repeatedly, became increasingly specific, and was eventually stored as if the user had said it. Extraction now receives only user turns, prompts forbid plausible additions, and a regression test verifies that an invented assistant detail never reaches the extractor prompt.
- **Buffer-only extraction caused silent end-of-session loss.** A late correction remained only in the live buffer and vanished when the process exited. `Session.flush()` now extracts remaining user facts on close.
- **Buffer size affects reliability as well as speed.** An artificially small test budget forced repeated re-summarization, increasing both latency and drift exposure. The live test budget was raised from 350 to 600 tokens.
- **Representative small evaluation:** nine of ten facts were recalled across two scripted dialogs, both contradiction cases passed, and the buffer remained within budget. Generation is nondeterministic, so this is an observed run rather than a fixed guarantee.

See [`readme.html`](readme.html) for the architecture, contradiction calibration, audit trail, and demo transcript.
