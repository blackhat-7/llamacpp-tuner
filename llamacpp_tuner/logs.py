"""Read human prompts and subagent briefs from local coding-agent logs.

Sources: Claude Code, Pi, opencode and Codex, each under the user's home directory.
Each row is a dict: host, src, kind (human, delegate or subagent_first), project,
session, model, agent, text. Fine-tuning data is mined from these rows.
"""

import json
import socket
import sqlite3
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

# Text the harness injects into the user's turn, not something the user typed.
SKIP_PREFIXES = (
    "<command-",
    "<local-command",
    "<system-reminder",
    "Caveat:",
    "[Request interrupted",
    "<task-notification",
    "<bash-",
)


def _row(
    src: str,
    kind: str,
    project: str,
    session: str,
    text: Any,
    model: str | None = None,
    agent: str | None = None,
) -> dict | None:
    text = text.strip() if isinstance(text, str) else ""
    if not text or text.startswith(SKIP_PREFIXES):
        return None
    return {
        "host": socket.gethostname(),
        "src": src,
        "kind": kind,
        "project": project,
        "session": session,
        "model": model,
        "agent": agent,
        "text": text,
    }


def _texts(content: Any) -> list[str]:
    if isinstance(content, str):
        return [content]
    if not isinstance(content, list):
        return []
    return [
        p.get("text", "")
        for p in content
        if isinstance(p, dict) and p.get("type") == "text"
    ]


def _records(path: Path) -> Iterator[dict]:
    # Agents append while they run, so the last line may be cut short.
    with path.open(errors="replace") as lines:
        for line in lines:
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if isinstance(record, dict):
                yield record


def claude_code(home: Path) -> Iterator[dict]:
    root = home / ".claude" / "projects"
    for path in sorted(root.glob("**/*.jsonl")):
        project = path.relative_to(root).parts[0]
        is_subagent = "subagents" in path.relative_to(root).parts
        first = True
        for r in _records(path):
            msg = r.get("message") or {}
            if (
                r.get("type") == "user"
                and not r.get("isMeta")
                and not r.get("toolUseResult")
            ):
                for text in _texts(msg.get("content")):
                    if is_subagent or r.get("isSidechain"):
                        # Only a subagent's first message is its brief; later ones are tool chatter.
                        row = (
                            _row("claude", "subagent_first", project, str(path), text)
                            if first
                            else None
                        )
                    else:
                        row = _row("claude", "human", project, str(path), text)
                    if row:
                        yield row
                first = False
            elif r.get("type") == "assistant":
                for part in msg.get("content") or []:
                    if (
                        isinstance(part, dict)
                        and part.get("type") == "tool_use"
                        and part.get("name") in ("Agent", "Task")
                    ):
                        args = part.get("input") or {}
                        row = _row(
                            "claude",
                            "delegate",
                            project,
                            str(path),
                            args.get("prompt"),
                            model=args.get("model"),
                            agent=args.get("subagent_type"),
                        )
                        if row:
                            yield row


def pi(home: Path) -> Iterator[dict]:
    root = home / ".pi" / "agent" / "sessions"
    for path in sorted(root.glob("**/*.jsonl")):
        project = path.relative_to(root).parts[0]
        for r in _records(path):
            msg = r.get("message") or {}
            if r.get("type") == "message" and msg.get("role") == "user":
                for text in _texts(msg.get("content")):
                    if row := _row("pi", "human", project, str(path), text):
                        yield row


def opencode(home: Path) -> Iterator[dict]:
    db = home / ".local" / "share" / "opencode" / "opencode.db"
    if not db.exists():
        return
    query = """select s.directory, s.id, s.parent_id, m.data, p.data from part p
               join message m on m.id = p.message_id join session s on s.id = p.session_id
               order by p.time_created"""
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        briefed = set()
        for directory, sid, parent, mdata, pdata in con.execute(query):
            m, p = json.loads(mdata), json.loads(pdata)
            if m.get("role") != "user" or p.get("type") != "text" or p.get("synthetic"):
                continue
            model = (m.get("model") or {}).get("modelID")
            if not parent:
                row = _row(
                    "opencode",
                    "human",
                    directory,
                    sid,
                    p.get("text"),
                    model=model,
                    agent=m.get("agent"),
                )
            elif sid not in briefed:
                briefed.add(sid)
                row = _row(
                    "opencode",
                    "subagent_first",
                    directory,
                    sid,
                    p.get("text"),
                    model=model,
                    agent=m.get("agent"),
                )
            else:
                continue
            if row:
                yield row
    finally:
        con.close()


def codex(home: Path) -> Iterator[dict]:
    for path in sorted((home / ".codex" / "sessions").glob("**/*.jsonl")):
        for r in _records(path):
            payload = r.get("payload") or {}
            if (
                r.get("type") == "event_msg"
                and payload.get("type") == "user_message"
                and (
                    row := _row("codex", "human", "", str(path), payload.get("message"))
                )
            ):
                yield row


SOURCES: dict[str, Callable[[Path], Iterator[dict]]] = {
    "claude": claude_code,
    "pi": pi,
    "opencode": opencode,
    "codex": codex,
}
