"""对话历史的持久化存储（SQLite，标准库，无额外依赖）。

LangGraph 的 MemorySaver 只活在内存里，服务一重启，所有会话的上下文就没了。
这里把每一轮问答落盘，供两件事用：

  1. 左侧「会话历史」列表的展示、切换、删除；
  2. agent 重建上下文（见 agent.py 的 _seed_from_history）。

数据文件：BASE_DIR/history.db
"""

import os
import sqlite3
import threading
import time
from contextlib import contextmanager, closing

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "history.db")

DEFAULT_TITLE = "新会话"
MAX_TITLE = 60

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id         TEXT PRIMARY KEY,
    title      TEXT NOT NULL DEFAULT '新会话',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    role       TEXT NOT NULL,
    content    TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id);
"""

_lock = threading.RLock()
_inited = False


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def _ensure() -> None:
    """建表（幂等）。多线程下只做一次。"""
    global _inited
    if _inited:
        return
    with _lock:
        if _inited:
            return
        with closing(_connect()) as conn, conn:
            conn.executescript(_SCHEMA)
        _inited = True


@contextmanager
def _db():
    """一个自动提交/回滚的短连接。

    每次调用新开连接：sqlite3 的连接不是线程安全的，而问答、上传、删除
    可能来自不同的线程（上传走 asyncio.to_thread）。本地文件的开销是微秒级。
    """
    _ensure()
    conn = _connect()
    try:
        yield conn
        conn.commit()
    except BaseException:
        conn.rollback()
        raise
    finally:
        conn.close()


def record_turn(session_id: str, question: str, answer: str) -> None:
    """记一轮问答：会话不存在就顺手建出来，用第一句提问当标题。"""
    if not session_id:
        return
    now = time.time()
    with _lock, _db() as conn:
        row = conn.execute(
            "SELECT id FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        if row is None:
            conn.execute(
                "INSERT INTO sessions (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (session_id, _title_from(question), now, now),
            )
        else:
            conn.execute(
                "UPDATE sessions SET updated_at = ? WHERE id = ?", (now, session_id)
            )
        if question:
            conn.execute(
                "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                (session_id, "user", question, now),
            )
        if answer:
            conn.execute(
                "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                (session_id, "assistant", answer, now),
            )


def _title_from(text: str) -> str:
    flat = " ".join((text or "").split())
    if not flat:
        return DEFAULT_TITLE
    return flat[:MAX_TITLE] + ("…" if len(flat) > MAX_TITLE else "")


def list_sessions(limit: int = 200) -> list[dict]:
    """按最近活动时间倒序列出会话。"""
    with _lock, _db() as conn:
        rows = conn.execute(
            "SELECT s.id, s.title, s.created_at, s.updated_at, "
            "  (SELECT COUNT(*) FROM messages m WHERE m.session_id = s.id) AS message_count "
            "FROM sessions s ORDER BY s.updated_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_session(session_id: str) -> dict | None:
    with _lock, _db() as conn:
        row = conn.execute(
            "SELECT id, title, created_at, updated_at FROM sessions WHERE id = ?",
            (session_id,),
        ).fetchone()
    return dict(row) if row else None


def get_messages(session_id: str) -> list[dict]:
    """按时间顺序返回一条会话的全部消息（给前端渲染）。"""
    with _lock, _db() as conn:
        rows = conn.execute(
            "SELECT role, content, created_at FROM messages "
            "WHERE session_id = ? ORDER BY id",
            (session_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def recent_messages(session_id: str, limit: int) -> list[dict]:
    """最近 limit 条消息，按时间正序返回（给模型垫上下文用）。"""
    with _lock, _db() as conn:
        rows = conn.execute(
            "SELECT role, content FROM messages WHERE session_id = ? "
            "ORDER BY id DESC LIMIT ?",
            (session_id, limit),
        ).fetchall()
    return [dict(r) for r in reversed(rows)]


def delete_session(session_id: str) -> int:
    """删掉一条会话及其全部消息，返回删掉的消息条数。"""
    with _lock, _db() as conn:
        cur = conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        removed = cur.rowcount or 0
        conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
    return removed


def rename_session(session_id: str, title: str) -> bool:
    with _lock, _db() as conn:
        cur = conn.execute(
            "UPDATE sessions SET title = ?, updated_at = ? WHERE id = ?",
            (_title_from(title), time.time(), session_id),
        )
        return bool(cur.rowcount)
