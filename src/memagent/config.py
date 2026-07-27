"""Central configuration: models, budgets, thresholds, and paths.

This is the ONE place model names are defined. Both the generation/
summarization/extraction model and the embedding model are served
locally by Ollama.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    # --- Ollama connection ---
    ollama_host: str = "http://localhost:11434"

    # --- Models (the ONE place these names live) ---
    generation_model: str = "qwen3.5:9b"
    embedding_model: str = "nomic-embed-text"

    # --- Storage ---
    project_root: Path = Path(__file__).resolve().parent.parent.parent
    data_dir: Path = project_root / "data"
    chroma_dir: Path = data_dir / "chroma_db"
    sqlite_path: Path = data_dir / "memagent.sqlite3"

    # --- Short-term buffer ---
    # Rolling window of conversation history kept verbatim, budgeted in
    # (approximate) tokens. Compression fires once the buffer reaches
    # `compression_trigger_ratio` of the budget, not once it overflows --
    # this leaves headroom so the context assembled for the *next* turn
    # (which still includes the pre-compression buffer) never blows past
    # budget either.
    buffer_budget_tokens: int = 2000
    compression_trigger_ratio: float = 0.8
    # Turns kept verbatim (uncompressed) after a compression pass.
    keep_last_n_turns: int = 4

    # --- Compression / extraction ---
    compression_temperature: float = 0.2
    extraction_temperature: float = 0.0

    # --- Long-term recall ---
    # score = similarity * exp(-age_days / half_life_days) * (importance / 5)
    recall_half_life_days: float = 30.0
    recall_top_k: int = 4
    # How many nearest-neighbor candidates to pull from Chroma before
    # rescoring with recency/importance and truncating to top_k.
    recall_candidate_k: int = 12

    # --- Dedup / contradiction handling ---
    # Cosine similarity above which a newly extracted fact is compared
    # against an existing one (LLM decides replace/merge/skip) rather than
    # inserted as a brand-new memory.
    #
    # Calibrated empirically against nomic-embed-text (see readme.html for
    # the full table): near-exact paraphrases of the same fact land ~0.90,
    # but genuine contradictions about the same "slot" -- the cases this
    # gate exists to catch -- score much lower than intuition suggests:
    # "lives in NYC" vs "moved to Boston" ~0.71-0.77, "uses Chroma" vs
    # "switched from Chroma to Qdrant" ~0.79. Unrelated facts cluster at
    # 0.39-0.44. A 0.9 threshold (a plausible-looking default) would miss
    # every contradiction case and only catch verbatim restatements. 0.65
    # sits with clean margin above the unrelated ceiling and below the
    # lowest observed contradiction score.
    dedup_similarity_threshold: float = 0.65
    dedup_temperature: float = 0.0

    # --- LLM call tuning ---
    # Ollama is shared with other builders; under load a single call has
    # been observed to take up to ~240s, so timeouts are generous.
    llm_timeout_s: float = 240.0
    llm_max_retries: int = 1
    chat_temperature: float = 0.4


CONFIG = Config()
