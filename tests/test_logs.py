"""Tests for exporting prompts and subagent briefs from coding-agent logs."""

import json
import sqlite3
from pathlib import Path

from click.testing import CliRunner

from llamacpp_tuner.cli import main


def write_jsonl(path: Path, records: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n{cut off")


def user(content: object, **extra: object) -> dict:
    return {"type": "user", "message": {"role": "user", "content": content}, **extra}


def make_claude(home: Path) -> None:
    project = home / ".claude" / "projects" / "-home-me-repo"
    write_jsonl(
        project / "s1.jsonl",
        [
            user("Fix the bug"),
            user("<system-reminder>injected</system-reminder>"),
            user("expanded skill text", isMeta=True),
            user([{"type": "tool_result", "content": "ok"}], toolUseResult={}),
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {"type": "text", "text": "Delegating"},
                        {
                            "type": "tool_use",
                            "name": "Agent",
                            "input": {
                                "prompt": "Find the caller",
                                "model": "haiku",
                                "subagent_type": "Explore",
                            },
                        },
                    ]
                },
            },
            user([{"type": "text", "text": "Now make it shorter"}]),
        ],
    )
    write_jsonl(
        project / "s1" / "subagents" / "agent-1.jsonl",
        [
            user("Find the caller", isSidechain=True),
            user("later turn", isSidechain=True),
        ],
    )


def make_pi(home: Path) -> None:
    write_jsonl(
        home / ".pi" / "agent" / "sessions" / "--home-me-repo--" / "a.jsonl",
        [
            {
                "type": "message",
                "message": {
                    "role": "user",
                    "content": [{"type": "text", "text": "Pi ask"}],
                },
            },
            {"type": "message", "message": {"role": "assistant", "content": "answer"}},
        ],
    )


def make_opencode(home: Path) -> None:
    db = home / ".local" / "share" / "opencode" / "opencode.db"
    db.parent.mkdir(parents=True)
    con = sqlite3.connect(db)
    con.executescript(
        """create table session (id text, parent_id text, directory text);
           create table message (id text, data text);
           create table part (id text, message_id text, session_id text, time_created int, data text);"""
    )
    con.executemany(
        "insert into session values (?, ?, ?)",
        [("root", None, "/repo"), ("child", "root", "/repo")],
    )
    messages = [
        (
            "m1",
            "root",
            {"role": "user", "model": {"modelID": "opus"}, "agent": "build"},
            {"type": "text", "text": "Open ask"},
        ),
        (
            "m2",
            "root",
            {"role": "user"},
            {"type": "text", "text": "added by the tool", "synthetic": True},
        ),
        ("m3", "root", {"role": "assistant"}, {"type": "text", "text": "reply"}),
        (
            "m4",
            "child",
            {"role": "user", "agent": "explore"},
            {"type": "text", "text": "Child brief"},
        ),
        ("m5", "child", {"role": "user"}, {"type": "text", "text": "Child follow-up"}),
    ]
    for t, (mid, sid, message, part) in enumerate(messages):
        con.execute("insert into message values (?, ?)", (mid, json.dumps(message)))
        con.execute(
            "insert into part values (?, ?, ?, ?, ?)",
            (f"p{t}", mid, sid, t, json.dumps(part)),
        )
    con.commit()
    con.close()


def make_codex(home: Path) -> None:
    write_jsonl(
        home / ".codex" / "sessions" / "2026" / "11" / "r.jsonl",
        [
            {
                "type": "event_msg",
                "payload": {"type": "user_message", "message": "Codex ask"},
            },
            {
                "type": "event_msg",
                "payload": {"type": "agent_message", "message": "reply"},
            },
        ],
    )


def export(home: Path, monkeypatch) -> tuple[list[dict], str]:
    monkeypatch.setenv("HOME", str(home))
    result = CliRunner().invoke(main, ["logs", "export"])
    assert result.exit_code == 0, result.output
    return [json.loads(line) for line in result.stdout.splitlines()], result.stderr


def test_export_reads_every_source(tmp_path, monkeypatch):
    for make in (make_claude, make_pi, make_opencode, make_codex):
        make(tmp_path)

    rows, errors = export(tmp_path, monkeypatch)

    assert errors == ""
    assert [(r["src"], r["kind"], r["text"]) for r in rows] == [
        ("claude", "subagent_first", "Find the caller"),
        ("claude", "human", "Fix the bug"),
        ("claude", "delegate", "Find the caller"),
        ("claude", "human", "Now make it shorter"),
        ("pi", "human", "Pi ask"),
        ("opencode", "human", "Open ask"),
        ("opencode", "subagent_first", "Child brief"),
        ("codex", "human", "Codex ask"),
    ]
    delegate = rows[2]
    assert (delegate["project"], delegate["model"], delegate["agent"]) == (
        "-home-me-repo",
        "haiku",
        "Explore",
    )
    assert (rows[5]["project"], rows[5]["model"], rows[5]["agent"]) == (
        "/repo",
        "opus",
        "build",
    )


def test_export_skips_a_broken_store_and_keeps_the_rest(tmp_path, monkeypatch):
    make_pi(tmp_path)
    db = tmp_path / ".local" / "share" / "opencode" / "opencode.db"
    db.parent.mkdir(parents=True)
    db.write_text("not a database")

    rows, errors = export(tmp_path, monkeypatch)

    assert [r["text"] for r in rows] == ["Pi ask"]
    assert "opencode" in errors


def test_export_with_no_logs_prints_nothing(tmp_path, monkeypatch):
    assert export(tmp_path, monkeypatch) == ([], "")
