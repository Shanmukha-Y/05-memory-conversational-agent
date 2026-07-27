#!/usr/bin/env python3
"""Scripted-dialog eval: for each dialog, run session A live (planting
facts across enough turns to force compression), start a *fresh* Session B
against the same store (simulating a process restart), and check whether
session B's answers actually reflect the planted facts.

Metrics:
  - fact-recall rate: planted facts correctly reflected in session B's answers
  - contradiction-update correctness: superseding fact wins, old one doesn't leak
  - context-size discipline: buffer never grows unbounded across the conversation

Usage:
  uv run python eval/run_eval.py                # run 1 dialog (default, fast)
  uv run python eval/run_eval.py --limit 3       # run all bundled dialogs
  uv run python eval/run_eval.py --limit 1 --dialog bob_startup
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rich.console import Console
from rich.table import Table

from memagent.audit import AuditLog
from memagent.buffer import Buffer
from memagent.memory_store import MemoryStore
from memagent.session import Session

console = Console()
DIALOGS_DIR = Path(__file__).resolve().parent / "scripted_dialogs"

# Small budget so the ~12-turn scripted conversations actually cross the
# compression trigger within the eval, without needing 50-turn scripts.
# See tests/test_integration.py for why this isn't smaller: too tight a
# budget causes repeated re-summarization of already-compressed summaries,
# which measurably drives the model to hallucinate plausible-sounding
# specifics that were never said.
EVAL_BUFFER_BUDGET_TOKENS = 600


@dataclass
class DialogResult:
    name: str
    recall_hits: int
    recall_total: int
    max_buffer_tokens: int
    compressions: int
    contradiction_pass: bool | None
    session_b_answers: list[str] = field(default_factory=list)


def _contains_any(text: str, needles: list[str]) -> bool:
    lowered = text.lower()
    return any(n.lower() in lowered for n in needles)


def run_dialog(dialog: dict, data_dir: Path) -> DialogResult:
    user_id = dialog["user_id"]
    chroma_dir = data_dir / "chroma"
    sqlite_path = data_dir / "eval.sqlite3"

    store = MemoryStore(chroma_dir=chroma_dir, sqlite_path=sqlite_path)
    audit = AuditLog(sqlite_path=sqlite_path)
    buffer_a = Buffer(budget_tokens=EVAL_BUFFER_BUDGET_TOKENS, trigger_ratio=0.8)
    session_a = Session(user_id=user_id, store=store, audit=audit, buffer=buffer_a)

    max_tokens = 0
    for turn in dialog["session_a_turns"]:
        session_a.handle_turn(turn)
        max_tokens = max(max_tokens, session_a.buffer.total_tokens())

    compressions = len(audit.events(user_id, "compression"))
    session_a.close()

    # Simulate a process restart: fresh Session, fresh (empty) Buffer,
    # same on-disk store/audit.
    store_b = MemoryStore(chroma_dir=chroma_dir, sqlite_path=sqlite_path)
    audit_b = AuditLog(sqlite_path=sqlite_path)
    session_b = Session(user_id=user_id, store=store_b, audit=audit_b, buffer=Buffer())

    hits = 0
    answers = []
    for q in dialog["session_b_questions"]:
        answer = session_b.handle_turn(q["question"])
        answers.append(answer)
        ok = _contains_any(answer, q["expect_any"])
        if ok and q.get("expect_none"):
            ok = not _contains_any(answer, q["expect_none"])
        hits += int(ok)

    contradiction_pass = None
    if "contradiction" in dialog:
        c = dialog["contradiction"]
        answer = session_b.handle_turn(c["question"])
        answers.append(answer)
        ok = _contains_any(answer, c["expect_any"])
        if ok and c.get("expect_none"):
            ok = not _contains_any(answer, c["expect_none"])
        contradiction_pass = ok

    session_b.close()

    return DialogResult(
        name=dialog["name"],
        recall_hits=hits,
        recall_total=len(dialog["session_b_questions"]),
        max_buffer_tokens=max_tokens,
        compressions=compressions,
        contradiction_pass=contradiction_pass,
        session_b_answers=answers,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=1, help="Number of dialogs to run (default: 1, keeps eval fast).")
    parser.add_argument("--dialog", type=str, default=None, help="Run only the dialog with this name.")
    parser.add_argument("--verbose", action="store_true", help="Print each session B answer.")
    args = parser.parse_args()

    dialog_files = sorted(DIALOGS_DIR.glob("*.json"))
    dialogs = [json.loads(p.read_text()) for p in dialog_files]
    if args.dialog:
        dialogs = [d for d in dialogs if d["name"] == args.dialog]
    else:
        dialogs = dialogs[: args.limit]

    if not dialogs:
        console.print("[red]No matching dialogs found.[/]")
        raise SystemExit(1)

    console.print(f"[bold]Running {len(dialogs)} scripted dialog(s)...[/]\n")

    data_dir = Path(tempfile.mkdtemp(prefix="memagent_eval_"))
    results: list[DialogResult] = []
    try:
        for dialog in dialogs:
            console.print(f"[cyan]-> {dialog['name']}[/]")
            result = run_dialog(dialog, data_dir / dialog["name"])
            results.append(result)
            if args.verbose:
                for q, a in zip(dialog["session_b_questions"], result.session_b_answers):
                    console.print(f"  [dim]Q: {q['question']}[/]\n  A: {a}\n")
    finally:
        shutil.rmtree(data_dir, ignore_errors=True)

    table = Table(title="Recall eval results")
    table.add_column("dialog")
    table.add_column("fact recall", justify="right")
    table.add_column("compressions", justify="right")
    table.add_column("max buffer tokens", justify="right")
    table.add_column("contradiction")
    for r in results:
        contradiction_str = "-" if r.contradiction_pass is None else ("PASS" if r.contradiction_pass else "FAIL")
        table.add_row(
            r.name,
            f"{r.recall_hits}/{r.recall_total}",
            str(r.compressions),
            f"{r.max_buffer_tokens} / {EVAL_BUFFER_BUDGET_TOKENS} budget",
            contradiction_str,
        )
    console.print(table)

    total_hits = sum(r.recall_hits for r in results)
    total_qs = sum(r.recall_total for r in results)
    contradiction_results = [r.contradiction_pass for r in results if r.contradiction_pass is not None]
    console.print(
        f"\n[bold]Aggregate recall rate:[/] {total_hits}/{total_qs} "
        f"({total_hits / total_qs:.0%})" if total_qs else "n/a"
    )
    if contradiction_results:
        passed = sum(contradiction_results)
        console.print(f"[bold]Contradiction handling:[/] {passed}/{len(contradiction_results)} passed")
    over_budget = [r for r in results if r.max_buffer_tokens > EVAL_BUFFER_BUDGET_TOKENS * 1.5]
    if over_budget:
        console.print(f"[red]Context discipline: {len(over_budget)} dialog(s) blew well past budget![/]")
    else:
        console.print("[green]Context discipline: buffer stayed bounded in all dialogs.[/]")


if __name__ == "__main__":
    main()
