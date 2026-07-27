"""`mem chat --user <id>` -- rich CLI for the memory-enabled agent.

Commands inside the chat loop:
  /memories        list distilled long-term facts for this user
  /forget <id>      deactivate a fact (id or unambiguous id prefix)
  /why              show which memories were recalled for the last reply,
                    and their similarity/recency/importance/score
  /help             list commands
  /exit, /quit      leave
"""

from __future__ import annotations

import click
from rich.console import Console
from rich.table import Table

from memagent.llm import LLMError
from memagent.session import Session

console = Console()


def _resolve_fact_id(session: Session, id_or_prefix: str) -> str | None:
    id_or_prefix = id_or_prefix.strip()
    facts = session.store.list_facts(session.user_id)
    exact = [f for f in facts if f.id == id_or_prefix]
    if exact:
        return exact[0].id
    matches = [f for f in facts if f.id.startswith(id_or_prefix)]
    if len(matches) == 1:
        return matches[0].id
    if len(matches) > 1:
        console.print(f"[yellow]Ambiguous id prefix '{id_or_prefix}' matches {len(matches)} memories.[/]")
    return None


def _show_memories(session: Session) -> None:
    facts = session.store.list_facts(session.user_id)
    if not facts:
        console.print("[dim]No long-term memories yet for this user.[/]")
        return
    table = Table(title=f"Memories for {session.user_id}")
    table.add_column("id", style="dim")
    table.add_column("type")
    table.add_column("importance", justify="right")
    table.add_column("text")
    table.add_column("updated")
    for f in facts:
        table.add_row(f.id[:8], f.type, str(f.importance), f.text, f.updated_at.strftime("%Y-%m-%d %H:%M UTC"))
    console.print(table)


def _show_why(session: Session) -> None:
    if not session.last_recall:
        console.print("[yellow]No long-term memories were recalled for the last reply.[/]")
        return
    table = Table(title="Recalled for the last reply")
    table.add_column("text")
    table.add_column("similarity", justify="right")
    table.add_column("age (days)", justify="right")
    table.add_column("decay", justify="right")
    table.add_column("importance", justify="right")
    table.add_column("score", justify="right")
    for r in session.last_recall:
        table.add_row(
            r.record.text,
            f"{r.similarity:.3f}",
            f"{r.age_days:.1f}",
            f"{r.decay:.3f}",
            str(r.record.importance),
            f"{r.score:.3f}",
        )
    console.print(table)


HELP_TEXT = """[bold]Commands[/]
  /memories        list distilled long-term facts for this user
  /forget <id>     deactivate a memory (id or unambiguous prefix)
  /why             show what was recalled for the last reply, and why
  /help            this message
  /exit, /quit     leave"""


def _handle_command(session: Session, raw: str) -> bool:
    """Returns True if the REPL should exit."""
    parts = raw.strip().split(maxsplit=1)
    name = parts[0].lower()
    arg = parts[1] if len(parts) > 1 else ""

    if name in ("/exit", "/quit"):
        return True
    if name == "/help":
        console.print(HELP_TEXT)
    elif name == "/memories":
        _show_memories(session)
    elif name == "/forget":
        if not arg:
            console.print("[red]usage: /forget <id>[/]")
        else:
            fact_id = _resolve_fact_id(session, arg)
            if fact_id and session.store.forget(session.user_id, fact_id):
                console.print(f"[green]forgot memory {fact_id[:8]}[/]")
            else:
                console.print(f"[red]no unique matching memory for '{arg}'[/]")
    elif name == "/why":
        _show_why(session)
    else:
        console.print(f"[red]unknown command {name!r} -- try /help[/]")
    return False


@click.group()
def cli() -> None:
    """mem -- memory-enabled conversational agent."""


@cli.command()
@click.option("--user", "user_id", required=True, help="User id to chat as (memory is keyed by this).")
def chat(user_id: str) -> None:
    """Start an interactive chat session for USER."""
    session = Session(user_id=user_id)
    console.print(
        f"[bold cyan]mem chat[/] -- user=[bold]{user_id}[/]. Type /help for commands, /exit to quit."
    )
    try:
        while True:
            try:
                user_input = console.input("[bold green]you>[/] ")
            except (EOFError, KeyboardInterrupt):
                console.print()
                break
            if not user_input.strip():
                continue
            if user_input.startswith("/"):
                if _handle_command(session, user_input):
                    break
                continue
            try:
                reply = session.handle_turn(user_input)
            except LLMError as exc:
                console.print(f"[red]LLM error: {exc}[/]")
                continue
            console.print(f"[bold magenta]agent>[/] {reply}")
    finally:
        session.close()


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
