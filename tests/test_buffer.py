"""Buffer token accounting and compression-trigger math. No LLM involved."""

from memagent.buffer import Buffer, Turn, estimate_tokens


def test_estimate_tokens_heuristic():
    assert estimate_tokens("") == 0
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("a" * 40) == 10


def test_total_tokens_sums_all_turns():
    buf = Buffer(budget_tokens=1000, trigger_ratio=0.8)
    buf.add(Turn(role="user", content="a" * 40))  # 10 tokens
    buf.add(Turn(role="assistant", content="b" * 80))  # 20 tokens
    assert buf.total_tokens() == 30


def test_trigger_threshold_is_ratio_of_budget():
    buf = Buffer(budget_tokens=2000, trigger_ratio=0.8)
    assert buf.trigger_threshold() == 1600


def test_not_over_budget_below_threshold():
    buf = Buffer(budget_tokens=2000, trigger_ratio=0.8)
    buf.add(Turn(role="user", content="x" * 4000))  # 1000 tokens < 1600
    assert not buf.is_over_budget()


def test_over_budget_at_exact_threshold():
    buf = Buffer(budget_tokens=2000, trigger_ratio=0.8)
    buf.add(Turn(role="user", content="x" * (1600 * 4)))  # exactly 1600 tokens
    assert buf.is_over_budget()


def test_over_budget_above_threshold():
    buf = Buffer(budget_tokens=2000, trigger_ratio=0.8)
    buf.add(Turn(role="user", content="x" * 7000))  # 1750 tokens > 1600
    assert buf.is_over_budget()


def test_render_maps_summary_role_to_assistant():
    buf = Buffer()
    buf.add(Turn(role="summary", content="[compressed] earlier chat", compressed=True))
    buf.add(Turn(role="user", content="hi"))
    rendered = buf.render()
    assert rendered == [
        {"role": "assistant", "content": "[compressed] earlier chat"},
        {"role": "user", "content": "hi"},
    ]


def test_render_preserves_order():
    buf = Buffer()
    buf.add_message("user", "one")
    buf.add_message("assistant", "two")
    buf.add_message("user", "three")
    rendered = buf.render()
    assert [m["content"] for m in rendered] == ["one", "two", "three"]
