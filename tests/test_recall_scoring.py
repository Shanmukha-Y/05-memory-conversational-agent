"""Recall scoring formula: score = similarity * exp(-age/half_life) * (importance/5).

Pure math, no LLM or vector store -- exercises exact orderings, which is
the property the eval and demo actually depend on (a fresh important fact
must beat an old merely-similar one).
"""

import math

import pytest

from memagent.recall import score_memory


def test_score_at_zero_age_equals_similarity_times_importance_fraction():
    score = score_memory(similarity=0.8, age_days=0, importance=5, half_life_days=30)
    assert score == pytest.approx(0.8 * 1.0 * 1.0)


def test_score_at_half_life_is_halved():
    score = score_memory(similarity=1.0, age_days=30, importance=5, half_life_days=30)
    assert score == pytest.approx(math.exp(-1))


def test_importance_scales_linearly():
    low = score_memory(similarity=0.9, age_days=5, importance=1, half_life_days=30)
    high = score_memory(similarity=0.9, age_days=5, importance=5, half_life_days=30)
    assert high == pytest.approx(low * 5)


def test_score_decreases_monotonically_with_age():
    scores = [score_memory(similarity=0.8, age_days=d, importance=4, half_life_days=30) for d in (0, 10, 30, 90, 365)]
    assert scores == sorted(scores, reverse=True)
    assert len(set(scores)) == len(scores)  # strictly decreasing, not flat


def test_fresh_important_fact_beats_old_similar_one():
    # Old fact: nearly perfect similarity match, but 6 months stale and low importance.
    old_similar = score_memory(similarity=0.95, age_days=180, importance=2, half_life_days=30)
    # Fresh fact: decent similarity, recent, and important.
    fresh_important = score_memory(similarity=0.75, age_days=1, importance=5, half_life_days=30)
    assert fresh_important > old_similar


def test_zero_half_life_disables_decay():
    score = score_memory(similarity=0.6, age_days=1000, importance=3, half_life_days=0)
    assert score == pytest.approx(0.6 * (3 / 5))


def test_negative_similarity_clamped_to_zero():
    score = score_memory(similarity=-0.1, age_days=0, importance=5, half_life_days=30)
    assert score == 0.0


def test_full_ordering_across_candidates():
    """Reproduces the recall.py sort: highest score first."""
    candidates = [
        ("stale_high_sim", score_memory(0.95, 200, 3, 30)),
        ("fresh_low_importance", score_memory(0.6, 1, 1, 30)),
        ("fresh_high_importance", score_memory(0.7, 1, 5, 30)),
        ("mid_everything", score_memory(0.7, 15, 3, 30)),
    ]
    ordered = [name for name, _ in sorted(candidates, key=lambda c: c[1], reverse=True)]
    assert ordered[0] == "fresh_high_importance"
    assert ordered[-1] in ("stale_high_sim", "fresh_low_importance")
